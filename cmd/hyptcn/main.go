package main

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/example/hypTcn/internal/orchestrator"
	"github.com/spf13/cobra"
)

var (
	socketPath    = "/tmp/hyptcn.sock"
	vmName        = "guest"
	targetAddress = uint64(0x1000)
	intervalMs    = 100
	mockMode      = false
	rootCmd       = newRootCmd()
)

func main() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func newRootCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "hyptcn",
		Short: "Orchestrate the hypervisor forensic scan",
		RunE:  runScan,
	}

	cmd.Flags().StringVar(&socketPath, "socket", socketPath, "path to analyzer Unix domain socket")
	cmd.Flags().StringVar(&vmName, "vm", vmName, "libvmi guest domain name")
	cmd.Flags().Uint64Var(&targetAddress, "address", targetAddress, "physical address to sample (non-mock)")
	cmd.Flags().IntVar(&intervalMs, "interval", intervalMs, "sampling interval in milliseconds")
	cmd.Flags().BoolVar(&mockMode, "mock", mockMode, "generate fake frames instead of calling libvmi")
	return cmd
}

func runScan(cmd *cobra.Command, _ []string) error {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	interval := time.Duration(intervalMs) * time.Millisecond
	logger.Info("starting scan", "socket", socketPath, "vm", vmName, "address", targetAddress, "interval", interval, "mock", mockMode)

	engine := orchestrator.NewEngine(socketPath, vmName, mockMode, logger)

	if err := engine.Stream(ctx, targetAddress, interval); err != nil && !errors.Is(err, context.Canceled) {
		logger.Error("streaming aborted", "err", err)
		return err
	}

	logger.Info("stopping scan loop")
	return nil
}
