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
	"os"
	"path/filepath"
	"time"

	"github.com/example/hypTcn/internal/extractor"
	"log/slog"
)

const (
	dialTimeout           = 5 * time.Second
	writeTimeout          = 10 * time.Second
	readTimeout           = 10 * time.Second
	defaultSampleInterval = 5 * time.Second
	defaultProcInterval   = 100

	normMaxMods  = 200.0
	normMaxConns = 100.0

	alertThreshold = 0.85

	// collectProgressEvery is how many frames between progress log lines.
	collectProgressEvery = 100
)

// collectLabelIDs maps label name → uint32 label_id written into .bin files.
// normal=0, malware=1, shellcode=2, rootkit=3, cryptominer=4, ransomware=5
var collectLabelIDs = map[string]uint32{
	"normal":      0,
	"malware":     1,
	"shellcode":   2,
	"rootkit":     3,
	"cryptominer": 4,
	"ransomware":  5,
}

// Config holds all engine parameters.
type Config struct {
	// VMI
	SocketPath   string
	VMName       string
	SysmapPath   string
	Mock         bool
	ProcInterval int

	// Collection mode
	CollectMode     bool
	CollectLabel    string
	CollectDir      string
	CollectDuration time.Duration // 0 = unlimited

	// Output
	JSONOutput bool // print scores as JSON to stdout
	Quiet      bool // suppress stdout JSON (overrides JSONOutput)
}

// prediction is the JSON response emitted by the Python analyzer per frame.
type prediction struct {
	AnomalyScore  float64 `json:"anomaly_score"`
	Status        string  `json:"status"`
	ActivityClass string  `json:"activity_class,omitempty"`
}

// liveScore is what the Go binary prints to stdout when JSONOutput is set.
type liveScore struct {
	Addr          string  `json:"addr"`
	AnomalyScore  float64 `json:"anomaly_score"`
	Alert         bool    `json:"alert"`
	ActivityClass string  `json:"activity_class,omitempty"`
	Status        string  `json:"status"`
}

// Engine manages the lifecycle of analysis requests via the analyzer service.
type Engine struct {
	cfg          Config
	nextMockAddr uint64
	logger       *slog.Logger
	conn         net.Conn
	reader       *bufio.Reader
	vmiHandle    *extractor.Handle
	osLayerActive bool

	// Frame counter (for collect progress and OS-layer scan scheduling).
	frameCount int

	// OS-layer telemetry (updated every cfg.ProcInterval frames).
	lastProcCount     int
	lastModCountNorm  float32
	lastConnCountNorm float32

	// Collection state.
	collectStart time.Time
}

// NewEngine builds a reusable Engine.
func NewEngine(cfg Config, logger *slog.Logger) *Engine {
	if logger == nil {
		logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	if cfg.ProcInterval <= 0 {
		cfg.ProcInterval = defaultProcInterval
	}
	return &Engine{
		cfg:          cfg,
		nextMockAddr: 0x1000,
		logger:       logger,
	}
}

// Stream opens the VMI handle (non-mock, non-collect mode), continuously captures
// pages, and forwards them to the analyzer and/or writes .bin files.
//
// Wire format sent to Python (Go→Python per frame, 4112 bytes):
//
//	[8 bytes:  uint64 LE physical address        ]
//	[4 bytes:  float32 LE kernel_module_count_norm]
//	[4 bytes:  float32 LE network_conn_count_norm ]
//	[4096 bytes: raw page data                    ]
func (e *Engine) Stream(ctx context.Context, physicalAddress uint64, interval time.Duration) error {
	if interval <= 0 {
		interval = defaultSampleInterval
	}

	// In collection mode with a duration limit, wrap the context.
	if e.cfg.CollectMode && e.cfg.CollectDuration > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, e.cfg.CollectDuration)
		defer cancel()
	}

	// Open VMI handle (non-mock mode).
	if !e.cfg.Mock {
		var h *extractor.Handle
		var err error
		if e.cfg.SysmapPath != "" {
			h, err = extractor.OpenWithSysmap(e.cfg.VMName, e.cfg.SysmapPath)
			if err != nil {
				e.logger.Warn("OpenWithSysmap failed, trying raw mode", "err", err)
				h, err = extractor.Open(e.cfg.VMName)
			}
		} else {
			h, err = extractor.Open(e.cfg.VMName)
		}
		if err != nil {
			return fmt.Errorf("open vmi: %w", err)
		}
		e.vmiHandle = h
		e.osLayerActive = h.HasOSLayer()
		e.logger.Info("VMI opened", "os_layer", e.osLayerActive)
	}

	defer e.Close()

	// Collection book-keeping (runs after Close() due to LIFO defer order).
	if e.cfg.CollectMode {
		e.collectStart = time.Now()
		defer e.printCollectStats()
	}

	// Initial capture.
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

	// Refresh OS-layer telemetry on schedule.
	if e.osLayerActive && e.frameCount%e.cfg.ProcInterval == 0 {
		e.refreshOSMetrics()
	}

	// Acquire page.
	var addr uint64
	var payload []byte

	if e.cfg.Mock {
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

	// In collection mode: write .bin file and show progress — skip Python.
	if e.cfg.CollectMode {
		if err := e.writeBinFrame(addr, payload); err != nil {
			e.logger.Error("failed to write .bin frame", "err", err)
		}
		if e.frameCount%collectProgressEvery == 0 {
			e.printCollectProgress()
		}
		return nil
	}

	// Normal analysis mode: send to Python analyzer.
	e.logger.Debug("dispatching payload", "bytes", len(payload), "address", addr)
	pred, err := e.analyze(ctx, addr, payload)
	if err != nil {
		return fmt.Errorf("analyzer: %w", err)
	}

	// Optionally print JSON score to stdout.
	if e.cfg.JSONOutput && !e.cfg.Quiet && pred.Status == "ok" {
		score := liveScore{
			Addr:          fmt.Sprintf("0x%x", addr),
			AnomalyScore:  pred.AnomalyScore,
			Alert:         pred.AnomalyScore > alertThreshold,
			ActivityClass: pred.ActivityClass,
			Status:        pred.Status,
		}
		if data, err := json.Marshal(score); err == nil {
			fmt.Println(string(data))
		}
	}

	return nil
}

// writeBinFrame writes one frame to <CollectDir>/<CollectLabel>/<frameCount>_<addr>.bin.
//
// Binary format (4108 bytes):
//
//	[8 bytes:  uint64 LE physical address]
//	[4 bytes:  uint32 LE label_id        ]
//	[4096 bytes: raw page data           ]
func (e *Engine) writeBinFrame(addr uint64, payload []byte) error {
	dir := filepath.Join(e.cfg.CollectDir, e.cfg.CollectLabel)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", dir, err)
	}

	labelID, ok := collectLabelIDs[e.cfg.CollectLabel]
	if !ok {
		labelID = 1 // unknown → treat as generic malware
	}

	fname := fmt.Sprintf("%08d_%016x.bin", e.frameCount, addr)
	fpath := filepath.Join(dir, fname)

	f, err := os.Create(fpath)
	if err != nil {
		return fmt.Errorf("create %s: %w", fpath, err)
	}
	defer f.Close()

	var hdr [12]byte
	binary.LittleEndian.PutUint64(hdr[0:8], addr)
	binary.LittleEndian.PutUint32(hdr[8:12], labelID)
	if _, err := f.Write(hdr[:]); err != nil {
		return err
	}
	_, err = f.Write(payload)
	return err
}

func (e *Engine) printCollectProgress() {
	elapsed := time.Since(e.collectStart)
	var remaining time.Duration
	if e.cfg.CollectDuration > 0 {
		remaining = e.cfg.CollectDuration - elapsed
		if remaining < 0 {
			remaining = 0
		}
	}
	fmt.Printf("[collect] %d frames | label=%s | elapsed=%ds | remaining=%ds\n",
		e.frameCount, e.cfg.CollectLabel,
		int(elapsed.Seconds()), int(remaining.Seconds()))
}

func (e *Engine) printCollectStats() {
	sequences := 0
	if e.frameCount >= 16 {
		sequences = (e.frameCount-16)/8 + 1
	}
	fmt.Printf("[collect] DONE | %d frames | label=%s | saved to %s%s/\n",
		e.frameCount, e.cfg.CollectLabel, e.cfg.CollectDir, e.cfg.CollectLabel)
	fmt.Printf("[collect] Estimated training sequences: ~%d (stride=8, window=16)\n", sequences)
}

// analyze sends one framed page to the Python analyzer service and returns its prediction.
func (e *Engine) analyze(ctx context.Context, address uint64, payload []byte) (*prediction, error) {
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

// refreshOSMetrics queries the guest OS for process, module, and connection counts.
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
		conn, err := dialer.DialContext(ctx, "unix", e.cfg.SocketPath)
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
		e.cfg.SocketPath, maxConnAttempts, lastErr)
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
