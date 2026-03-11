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
)

type prediction struct {
	AnomalyScore float64 `json:"anomaly_score"`
	Status       string  `json:"status"`
}

// Engine manages the lifecycle of analysis requests via the analyzer service.
type Engine struct {
	socketPath   string
	vmName       string
	mock         bool
	nextMockAddr uint64
	logger       *slog.Logger
	conn         net.Conn
	reader       *bufio.Reader
	vmiHandle    *extractor.Handle
}

// NewEngine builds a reusable Engine bound to a Unix domain socket.
// When mock is true the engine generates synthetic frames instead of calling libvmi.
func NewEngine(socketPath, vmName string, mock bool, logger *slog.Logger) *Engine {
	if logger == nil {
		logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}

	return &Engine{
		socketPath:   socketPath,
		vmName:       vmName,
		mock:         mock,
		nextMockAddr: 0x1000,
		logger:       logger,
	}
}

// Analyze streams one framed page to the analyzer service and returns its prediction.
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

	var header [8]byte
	binary.LittleEndian.PutUint64(header[:], address)
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

	e.logger.Info("analysis complete", "score", pred.AnomalyScore, "status", pred.Status)
	return &pred, nil
}

// Stream opens the VMI handle (non-mock mode), continuously captures pages, and
// forwards them to the analyzer. Closes both VMI handle and UDS connection on return.
func (e *Engine) Stream(ctx context.Context, physicalAddress uint64, interval time.Duration) error {
	if interval <= 0 {
		interval = defaultSampleInterval
	}

	if !e.mock {
		h, err := extractor.Open(e.vmName)
		if err != nil {
			return fmt.Errorf("open vmi: %w", err)
		}
		e.vmiHandle = h
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
		e.logger.Warn("analyzer socket unavailable, retrying", "attempt", attempt, "of", maxConnAttempts, "err", err)
		if attempt < maxConnAttempts {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(time.Second):
			}
		}
	}

	return fmt.Errorf("failed to dial analyzer socket %q after %d attempts: %w", e.socketPath, maxConnAttempts, lastErr)
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
