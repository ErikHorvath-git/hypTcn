package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/example/hypTcn/internal/orchestrator"
	"github.com/spf13/cobra"
)

// ── flag variables ─────────────────────────────────────────────────────────────

var (
	// VMI
	flagVM           string
	flagSocket       string
	flagSysmap       string
	flagAddress      uint64
	flagIntervalMs   int
	flagProcInterval int

	// Mock
	flagMock bool

	// Collection
	flagCollect         bool
	flagCollectLabel    string
	flagCollectDuration int
	flagCollectDir      string

	// Output
	flagLogLevel string
	flagJSON     bool
	flagQuiet    bool

	rootCmd = newRootCmd()
)

func main() {
	// Intercept --help / -h before cobra's flag parser so we can display
	// the grouped help without triggering stdlib flag.ErrHelp.
	for _, arg := range os.Args[1:] {
		if arg == "-h" || arg == "-help" || arg == "--help" {
			printGroupedHelp(os.Stdout)
			return
		}
	}
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func newRootCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "hyptcn",
		Short: "Hypervisor-aware security toolkit for live memory introspection on KVM guests",
		RunE:  runScan,
	}

	// ── VMI flags ─────────────────────────────────────────────────────────────
	cmd.Flags().StringVar(&flagVM, "vm", "",
		"KVM domain name")
	cmd.Flags().StringVar(&flagSocket, "socket", "/tmp/hyptcn.sock",
		"analyzer Unix domain socket path")
	cmd.Flags().StringVar(&flagSysmap, "sysmap", "",
		"System.map path for OS-layer introspection (enables vmi_init_complete)")
	cmd.Flags().Uint64Var(&flagAddress, "address", 0x1000000,
		"physical address to sample in non-mock mode")
	cmd.Flags().IntVar(&flagIntervalMs, "interval", 100,
		"sampling interval in milliseconds")
	cmd.Flags().IntVar(&flagProcInterval, "proc-interval", 100,
		"process/module/connection scan every N frames")

	// ── Mock flag ─────────────────────────────────────────────────────────────
	cmd.Flags().BoolVar(&flagMock, "mock", false,
		"generate synthetic random pages instead of calling libvmi")

	// ── Collection flags ──────────────────────────────────────────────────────
	cmd.Flags().BoolVar(&flagCollect, "collect", false,
		"enable dataset collection mode (writes .bin frames to --collect-dir)")
	cmd.Flags().StringVar(&flagCollectLabel, "collect-label", "normal",
		"label for collected frames: normal|malware|shellcode|rootkit|cryptominer|ransomware")
	cmd.Flags().IntVar(&flagCollectDuration, "collect-duration", 300,
		"collection duration in seconds (0 = run until Ctrl-C)")
	cmd.Flags().StringVar(&flagCollectDir, "collect-dir", "collect_for_training/",
		"root output directory for .bin frame files")

	// ── Output flags ──────────────────────────────────────────────────────────
	cmd.Flags().StringVar(&flagLogLevel, "log-level", "info",
		"structured log level: debug|info|warn|error")
	cmd.Flags().BoolVar(&flagJSON, "json", true,
		"print analysis results as JSON lines to stdout (use --json=false to disable)")
	cmd.Flags().BoolVar(&flagQuiet, "quiet", false,
		"suppress all stdout JSON output (overrides --json)")

	return cmd
}

// printGroupedHelp writes the grouped flag reference to w.
func printGroupedHelp(w io.Writer) {
	fmt.Fprintf(w, "hyptcn — hypervisor-aware security toolkit for live memory introspection\n\n")
	fmt.Fprintf(w, "Usage:\n  hyptcn [flags]\n\n")

	fmt.Fprintf(w, "VMI Flags:\n")
	fmt.Fprintf(w, "  --vm <name>              KVM domain name (default \"\")\n")
	fmt.Fprintf(w, "  --socket <path>          analyzer Unix domain socket (default \"/tmp/hyptcn.sock\")\n")
	fmt.Fprintf(w, "  --sysmap <path>          System.map for OS-layer introspection (default \"\")\n")
	fmt.Fprintf(w, "  --address <hex>          physical address to sample (default 0x1000000)\n")
	fmt.Fprintf(w, "  --interval <ms>          sampling interval in milliseconds (default 100)\n")
	fmt.Fprintf(w, "  --proc-interval <n>      OS-layer scan every N frames (default 100)\n")

	fmt.Fprintf(w, "\nMock Flags:\n")
	fmt.Fprintf(w, "  --mock                   generate synthetic pages instead of calling libvmi\n")

	fmt.Fprintf(w, "\nCollection Flags:\n")
	fmt.Fprintf(w, "  --collect                enable dataset collection mode\n")
	fmt.Fprintf(w, "  --collect-label <label>  normal|malware|shellcode|rootkit|cryptominer|ransomware (default \"normal\")\n")
	fmt.Fprintf(w, "  --collect-duration <sec> collection duration in seconds; 0 = until Ctrl-C (default 300)\n")
	fmt.Fprintf(w, "  --collect-dir <path>     output directory for .bin frames (default \"collect_for_training/\")\n")

	fmt.Fprintf(w, "\nOutput Flags:\n")
	fmt.Fprintf(w, "  --log-level <level>      debug|info|warn|error (default \"info\")\n")
	fmt.Fprintf(w, "  --json                   print scores as JSON to stdout (default true; use --json=false to disable)\n")
	fmt.Fprintf(w, "  --quiet                  suppress all stdout JSON output (overrides --json)\n")

	fmt.Fprintf(w, "\nExamples:\n")
	fmt.Fprintf(w, "  # Mock mode — no KVM required\n")
	fmt.Fprintf(w, "  hyptcn --mock --interval 100\n\n")
	fmt.Fprintf(w, "  # Live VM with OS-layer introspection\n")
	fmt.Fprintf(w, "  hyptcn --vm hyptcn-guest --sysmap configs/hyptcn-guest.sysmap --interval 500\n\n")
	fmt.Fprintf(w, "  # Collect normal behavior dataset for 5 minutes\n")
	fmt.Fprintf(w, "  hyptcn --mock --collect --collect-label normal --collect-duration 300\n")
}

func runScan(cmd *cobra.Command, _ []string) error {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	// Parse log level.
	var logLevel slog.Level
	switch flagLogLevel {
	case "debug":
		logLevel = slog.LevelDebug
	case "warn":
		logLevel = slog.LevelWarn
	case "error":
		logLevel = slog.LevelError
	default:
		logLevel = slog.LevelInfo
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: logLevel}))

	interval := time.Duration(flagIntervalMs) * time.Millisecond
	collectDur := time.Duration(flagCollectDuration) * time.Second

	logger.Info("starting hyptcn",
		"vm", flagVM,
		"sysmap", flagSysmap,
		"address", fmt.Sprintf("0x%x", flagAddress),
		"interval", interval,
		"mock", flagMock,
		"collect", flagCollect,
		"collect_label", flagCollectLabel,
		"collect_duration", collectDur,
	)

	cfg := orchestrator.Config{
		SocketPath:      flagSocket,
		VMName:          flagVM,
		SysmapPath:      flagSysmap,
		Mock:            flagMock,
		ProcInterval:    flagProcInterval,
		CollectMode:     flagCollect,
		CollectLabel:    flagCollectLabel,
		CollectDir:      flagCollectDir,
		CollectDuration: collectDur,
		JSONOutput:      flagJSON,
		Quiet:           flagQuiet,
	}

	engine := orchestrator.NewEngine(cfg, logger)

	err := engine.Stream(ctx, flagAddress, interval)
	if err != nil &&
		!errors.Is(err, context.Canceled) &&
		!errors.Is(err, context.DeadlineExceeded) {
		logger.Error("streaming aborted", "err", err)
		return err
	}

	logger.Info("scan stopped")
	return nil
}
