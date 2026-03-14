package orchestrator

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"time"

	"github.com/example/hypTcn/internal/extractor"
	"log/slog"
)

const (
	dialTimeout           = 5 * time.Second
	writeTimeout          = 10 * time.Second
	readTimeout           = 10 * time.Second
	defaultSampleInterval = 5 * time.Second

	// defaultProcInterval is the number of frames between OS-layer scans
	// (process list, kernel modules, network connections).
	defaultProcInterval = 100

	// normMaxProcs is the divisor used to normalise raw process count → [0,1].
	normMaxProcs = 300.0
	// normMaxMods is the divisor for kernel module count → [0,1].
	normMaxMods = 200.0
	// normMaxConns is the divisor for TCP connection count → [0,1].
	normMaxConns = 100.0
)

// prediction is the JSON response emitted by the Python analyzer per frame.
type prediction struct {
	AnomalyScore  float64 `json:"anomaly_score"`
	Status        string  `json:"status"`
	ActivityClass string  `json:"activity_class,omitempty"`
}

// Engine manages the lifecycle of analysis requests via the analyzer service.
type Engine struct {
	socketPath   string
	vmName       string
	sysmapPath   string
	mock         bool
	procInterval int
	nextMockAddr uint64
	logger       *slog.Logger
	conn         net.Conn
	reader       *bufio.Reader
	vmiHandle    *extractor.Handle
	osLayerActive bool

	// Running counters for OS-layer telemetry (updated every procInterval frames).
	frameCount       int
	lastProcCount    int
	lastModCountNorm  float32
	lastConnCountNorm float32
}

// NewEngine builds a reusable Engine bound to a Unix domain socket.
// sysmapPath may be empty (disables OS-layer init).
// procInterval sets how many frames elapse between OS-layer scans (0 → default 100).
// When mock is true the engine generates synthetic frames instead of calling libvmi.
func NewEngine(socketPath, vmName, sysmapPath string, mock bool, procInterval int, logger *slog.Logger) *Engine {
	if logger == nil {
		logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	if procInterval <= 0 {
		procInterval = defaultProcInterval
	}
	return &Engine{
		socketPath:   socketPath,
		vmName:       vmName,
		sysmapPath:   sysmapPath,
		mock:         mock,
		procInterval: procInterval,
		nextMockAddr: 0x1000,
		logger:       logger,
	}
}

// Analyze streams one framed page to the analyzer service and returns its prediction.
//
// Frame wire format (Go → Python):
//
//	[8 bytes:  uint64 LE physical address ]
//	[4 bytes:  float32 LE kernel_module_count_norm ]
//	[4 bytes:  float32 LE network_conn_count_norm  ]
//	[4096 bytes: raw page data                      ]
//
// Total: 4112 bytes per frame.
func (e *Engine) Analyze(ctx context.Context, address uint64, payload []byte) (*prediction, error) {
	if len(payload) == 0 {
		return nil, fmt.Errorf("payload must not be empty")
	}

	if err := e.connect(ctx); err != nil {
		return nil, err
	}

	if err := e.applyWriteDeadline(ctx); err != nil {
		return nil, err
	}

	// 16-byte header: address (8) + mod_count_norm (4) + conn_count_norm (4)
	var header [16]byte
	binary.LittleEndian.PutUint64(header[0:8], address)
	binary.LittleEndian.PutUint32(header[8:12], math.Float32bits(e.lastModCountNorm))
	binary.LittleEndian.PutUint32(header[12:16], math.Float32bits(e.lastConnCountNorm))

	if err := writeAll(e.conn, header[:]); err != nil {
		_ = e.closeConn()
		return nil, fmt.Errorf("failed to write header: %w", err)
	}
	if err := writeAll(e.conn, payload); err != nil {
		_ = e.closeConn()
		return nil, fmt.Errorf("failed to write payload: %w", err)
	}

	if err := e.applyReadDeadline(ctx); err != nil {
		return nil, err
	}

	line, err := e.reader.ReadBytes('\n')
	if err != nil {
		_ = e.closeConn()
		return nil, fmt.Errorf("failed to read analyzer response: %w", err)
	}

	var pred prediction
	if err := json.Unmarshal(bytes.TrimSpace(line), &pred); err != nil {
		return nil, fmt.Errorf("failed to decode analyzer response: %w", err)
	}

	e.logger.Info("analysis complete",
		"score", pred.AnomalyScore,
		"status", pred.Status,
		"activity_class", pred.ActivityClass,
	)
	return &pred, nil
}

// Stream opens the VMI handle (non-mock mode), continuously captures pages, and
// forwards them to the analyzer. Closes both VMI handle and UDS connection on return.
func (e *Engine) Stream(ctx context.Context, physicalAddress uint64, interval time.Duration) error {
	if interval <= 0 {
		interval = defaultSampleInterval
	}

	if !e.mock {
		var h *extractor.Handle
		var err error

		if e.sysmapPath != "" {
			h, err = extractor.OpenWithSysmap(e.vmName, e.sysmapPath)
			if err != nil {
				e.logger.Warn("OpenWithSysmap failed, trying raw mode", "err", err)
				h, err = extractor.Open(e.vmName)
			}
		} else {
			h, err = extractor.Open(e.vmName)
		}
		if err != nil {
			return fmt.Errorf("open vmi: %w", err)
		}

		e.vmiHandle = h
		e.osLayerActive = h.HasOSLayer()
		e.logger.Info("VMI opened",
			"os_layer", e.osLayerActive,
			"sysmap", e.sysmapPath != "",
		)
	}

	defer e.Close()

	if err := e.captureAndAnalyze(ctx, physicalAddress); err != nil {
		e.logger.Warn("initial capture failed", "err", err)
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := e.captureAndAnalyze(ctx, physicalAddress); err != nil {
				e.logger.Error("capture or analysis failed", "err", err)
			}
		}
	}
}

func (e *Engine) captureAndAnalyze(ctx context.Context, physicalAddress uint64) error {
	e.frameCount++

	// Every procInterval frames, refresh OS-layer telemetry.
	if e.osLayerActive && e.frameCount%e.procInterval == 0 {
		e.refreshOSMetrics()
	}

	var addr uint64
	var payload []byte

	if e.mock {
		addr = e.nextMockAddr
		e.nextMockAddr += 0x1000
		payload = make([]byte, extractor.PageSize)
		if _, err := rand.Read(payload); err != nil {
			return fmt.Errorf("generate mock page: %w", err)
		}
	} else {
		addr = physicalAddress
		var err error
		payload, err = e.vmiHandle.ReadPage(physicalAddress)
		if err != nil {
			return fmt.Errorf("read page: %w", err)
		}
	}

	e.logger.Debug("dispatching payload", "bytes", len(payload), "address", addr)
	if _, err := e.Analyze(ctx, addr, payload); err != nil {
		return fmt.Errorf("analyzer: %w", err)
	}

	return nil
}

// refreshOSMetrics queries the guest OS for process, module, and connection counts.
// Results are stored as normalized float32 values in the engine for inclusion
// in the next frame header.
func (e *Engine) refreshOSMetrics() {
	if e.vmiHandle == nil {
		return
	}

	procs, err := e.vmiHandle.GetProcessList()
	if err != nil {
		e.logger.Debug("GetProcessList failed", "err", err)
	} else {
		prev := e.lastProcCount
		e.lastProcCount = len(procs)
		if prev > 0 && e.lastProcCount < prev {
			e.logger.Warn("process count decreased — possible process hiding",
				"prev", prev, "now", e.lastProcCount)
		}
		e.logger.Debug("OS process scan", "count", e.lastProcCount)
	}

	mods, err := e.vmiHandle.GetKernelModules()
	if err != nil {
		e.logger.Debug("GetKernelModules failed", "err", err)
	} else {
		norm := float32(len(mods)) / normMaxMods
		if norm > 1.0 {
			norm = 1.0
		}
		e.lastModCountNorm = norm
		e.logger.Debug("OS module scan", "count", len(mods))
	}

	conns, err := e.vmiHandle.GetNetworkConnections()
	if err != nil {
		e.logger.Debug("GetNetworkConnections failed", "err", err)
	} else {
		norm := float32(len(conns)) / normMaxConns
		if norm > 1.0 {
			norm = 1.0
		}
		e.lastConnCountNorm = norm
		e.logger.Debug("OS connection scan", "count", len(conns))
	}
}

const maxConnAttempts = 3

func (e *Engine) connect(ctx context.Context) error {
	if e.conn != nil {
		return nil
	}

	dialer := &net.Dialer{Timeout: dialTimeout}
	var lastErr error
	for attempt := 1; attempt <= maxConnAttempts; attempt++ {
		conn, err := dialer.DialContext(ctx, "unix", e.socketPath)
		if err == nil {
			e.conn = conn
			e.reader = bufio.NewReader(conn)
			return nil
		}
		lastErr = err
		e.logger.Warn("analyzer socket unavailable, retrying",
			"attempt", attempt, "of", maxConnAttempts, "err", err)
		if attempt < maxConnAttempts {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(time.Second):
			}
		}
	}

	return fmt.Errorf("failed to dial analyzer socket %q after %d attempts: %w",
		e.socketPath, maxConnAttempts, lastErr)
}

// Close tears down the VMI handle and the persistent analyzer socket connection.
func (e *Engine) Close() error {
	if e.vmiHandle != nil {
		e.vmiHandle.Close()
		e.vmiHandle = nil
	}
	return e.closeConn()
}

func (e *Engine) closeConn() error {
	if e.conn == nil {
		return nil
	}
	err := e.conn.Close()
	e.conn = nil
	e.reader = nil
	if err != nil {
		return fmt.Errorf("close analyzer socket: %w", err)
	}
	return nil
}

func (e *Engine) applyWriteDeadline(ctx context.Context) error {
	if deadline, ok := ctx.Deadline(); ok {
		return e.conn.SetWriteDeadline(deadline)
	}
	return e.conn.SetWriteDeadline(time.Now().Add(writeTimeout))
}

func (e *Engine) applyReadDeadline(ctx context.Context) error {
	if deadline, ok := ctx.Deadline(); ok {
		return e.conn.SetReadDeadline(deadline)
	}
	return e.conn.SetReadDeadline(time.Now().Add(readTimeout))
}

func writeAll(conn net.Conn, data []byte) error {
	for len(data) > 0 {
		n, err := conn.Write(data)
		if err != nil {
			return err
		}
		data = data[n:]
	}
	return nil
}
