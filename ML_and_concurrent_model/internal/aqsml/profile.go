package aqsml

import (
	"fmt"
	"runtime"
	"time"
)

func profileStart(cfg Config, label string) time.Time {
	if cfg.Profile {
		fmt.Printf("[profile] start %-28s\n", label)
	}
	return time.Now()
}

func profileEnd(cfg Config, label string, start time.Time) {
	if !cfg.Profile {
		return
	}
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf(
		"[profile] end   %-28s elapsed=%8.3fs alloc=%8.1fMB sys=%8.1fMB num_gc=%d\n",
		label,
		time.Since(start).Seconds(),
		float64(m.Alloc)/1024.0/1024.0,
		float64(m.Sys)/1024.0/1024.0,
		m.NumGC,
	)
}
