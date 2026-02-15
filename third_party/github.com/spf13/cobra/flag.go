package cobra

import (
	"flag"
	"io"
	"time"
)

type FlagSet struct {
	fs *flag.FlagSet
}

func newFlagSet(name string) *FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	return &FlagSet{fs: fs}
}

func (f *FlagSet) StringVar(p *string, name, value, usage string) {
	f.fs.StringVar(p, name, value, usage)
}

func (f *FlagSet) Uint64Var(p *uint64, name string, value uint64, usage string) {
	f.fs.Uint64Var(p, name, value, usage)
}

func (f *FlagSet) DurationVar(p *time.Duration, name string, value time.Duration, usage string) {
	f.fs.DurationVar(p, name, value, usage)
}

func (f *FlagSet) Parse(args []string) error {
	return f.fs.Parse(args)
}

func (f *FlagSet) Args() []string {
	return f.fs.Args()
}
