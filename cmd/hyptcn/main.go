package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/example/hypTcn/internal/extractor"
	"github.com/example/hypTcn/internal/orchestrator"
	"github.com/spf13/cobra"
)

var (
	socketPath = "/tmp/hyptcn.sock"
	rootCmd    = newRootCmd()
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
	return cmd
}

func runScan(cmd *cobra.Command, _ []string) error {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	logger.Info("starting scan", "socket", socketPath)

	engine := orchestrator.NewEngine(socketPath, logger)

	dump, err := extractor.FetchPage(0x1000)
	if err != nil {
		logger.Error("failed to capture page", "err", err)
		return err
	}

	pred, err := engine.Analyze(ctx, dump)
	if err != nil {
		logger.Error("analysis failed", "err", err)
		return err
	}

	logger.Info("prediction", "score", pred.AnomalyScore, "status", pred.Status)
	return nil
}
