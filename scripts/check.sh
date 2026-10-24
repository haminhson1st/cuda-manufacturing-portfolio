#!/usr/bin/env bash
# Run one test command and print PASS/FAIL. Used by the Makefile test targets.
#   scripts/check.sh pass COMMAND [ARGS...]   expect exit status 0
#   scripts/check.sh fail COMMAND [ARGS...]   expect a clean error exit (status 1 or 2, not a crash)
set -u
expect="${1:?usage: check.sh pass|fail COMMAND [ARGS...]}"
shift
log="$(mktemp)"
"$@" > "$log" 2>&1
status=$?
case "$expect:$status" in
    pass:0 | fail:1 | fail:2)
        printf 'PASS  %s\n' "$*"
        rm -f "$log"
        exit 0
        ;;
esac
printf 'FAIL  %s  (exit %s; expected %s)\n' "$*" "$status" "$expect"
tail -n 40 "$log" | sed 's/^/      /'
rm -f "$log"
exit 1
