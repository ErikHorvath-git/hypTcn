package orchestrator

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"github.com/example/hypTcn/internal/extractor"
	"log/slog"
)

const (
	analyzeEndpoint       = "http://unix/analyze"
	dialTimeout           = 5 * time.Second
	defaultSampleInterval = 5 * time.Second
)

type prediction struct {
	AnomalyScore float64 `json:"anomaly_score"`
	Status       string  `json:"status"`
}

// Engine manages the lifecycle of analysis requests via the analyzer service.
type Engine struct {
	socketPath string
	vmName     string
	client     *http.Client
	logger     *slog.Logger
}

// NewEngine builds a reusable Engine bound to a Unix domain socket.
func NewEngine(socketPath, vmName string, logger *slog.Logger) *Engine {
	if logger == nil {
		logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}

	transport := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			dialer := &net.Dialer{Timeout: dialTimeout}
			return dialer.DialContext(ctx, "unix", socketPath)
		},
	}

	client := &http.Client{
		Transport: transport,
		Timeout:   10 * time.Second,
	}

	return &Engine{socketPath: socketPath, vmName: vmName, client: client, logger: logger}
}

// Analyze sends the in-memory dump to the analyzer service and returns its prediction.
func (e *Engine) Analyze(ctx context.Context, payload []byte) (*prediction, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, analyzeEndpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("failed to build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/octet-stream")

	e.logger.Debug("dispatching payload", "bytes", len(payload))

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("analyzer request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("analyzer returned %d: %s", resp.StatusCode, string(bytes.TrimSpace(body)))
	}

	var pred prediction
	if err := json.NewDecoder(resp.Body).Decode(&pred); err != nil {
		return nil, fmt.Errorf("failed to decode analyzer response: %w", err)
	}

	e.logger.Info("analysis complete", "score", pred.AnomalyScore, "status", pred.Status)
	return &pred, nil
}

// Stream continuously captures pages from the guest and forwards them to the analyzer.
func (e *Engine) Stream(ctx context.Context, physicalAddress uint64, interval time.Duration) error {
	if interval <= 0 {
		interval = defaultSampleInterval
	}

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
	payload, err := extractor.ExtractPage(e.vmName, physicalAddress)
	if err != nil {
		return fmt.Errorf("extract page: %w", err)
	}

	if _, err := e.Analyze(ctx, payload); err != nil {
		return fmt.Errorf("analyzer: %w", err)
	}

	return nil
}
