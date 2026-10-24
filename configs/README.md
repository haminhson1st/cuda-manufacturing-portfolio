# Frozen configurations

Each `.conf` file holds `name = value` lines, with `#` comments, for the options of [`inspect`](../tutorials/11-capstone.md) and [`release_bench`](../tutorials/12-portfolio-release.md). Command-line options override a file. A misspelled name fails the run, just like a misspelled option.

| File | Workflow | Origin |
| --- | --- | --- |
| [synthetic-images.conf](synthetic-images.conf) | `inspect images` on the synthetic Lab 07 set | Defaults of Lab 07; threshold 0 separates the synthetic set |
| [synthetic-sensors.conf](synthetic-sensors.conf) | `inspect sensors` on the synthetic Lab 04 recordings | Crest threshold used by the Lab 04 tests |

When you freeze a rule on real development data (Labs 04 and 07), save it here as, for example, `mvtec-bottle.conf` or `cwru-inner-race.conf`. Commit it before evaluating the held-out split, and cite its commit in the case study. Every `inspect` run copies the effective values into `REPORT.csv.json`, so a report can always be traced back to the exact rule that produced it.
