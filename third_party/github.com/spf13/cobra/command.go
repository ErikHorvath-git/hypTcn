package cobra

import "os"

type Command struct {
    Use             string
    Short           string
    RunE            func(cmd *Command, args []string) error
    flags           *FlagSet
    persistentFlags *FlagSet
}

func (c *Command) initFlags() {
    if c.flags == nil {
        c.flags = newFlagSet(c.Use)
    }
    if c.persistentFlags == nil {
        c.persistentFlags = newFlagSet(c.Use + "-persistent")
    }
}

func (c *Command) Flags() *FlagSet {
    c.initFlags()
    return c.flags
}

func (c *Command) PersistentFlags() *FlagSet {
    c.initFlags()
    return c.persistentFlags
}

func (c *Command) Execute() error {
    c.initFlags()
    args := os.Args[1:]
    if err := c.flags.Parse(args); err != nil {
        return err
    }
    if c.RunE == nil {
        return nil
    }
    return c.RunE(c, c.flags.Args())
}
