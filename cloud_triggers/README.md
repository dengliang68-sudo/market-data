# Public ETF option retry request

This directory contains no account, position, or credential data.
The bridge listens only to pushes on main changing cloud_triggers/opt-request.json.
Report commits do not match this path and cannot loop back into the bridge.

Request JSON must contain exactly:
```json
{"session":"opt","date":"YYYYMMDD","request_id":"opt-YYYYMMDD"}
```
Only session, a valid closed current XSHG trading date, and a nonempty safe request ID are allowed.
The bridge checks out the triggering commit SHA. JSON values are parsed as data, never inserted into shell code.
Normal requests are accepted from 20:35 Asia/Shanghai. A one-time migration acceptance
request is allowed earlier after the close on 2026-11-25 using request_id migration-acceptance-20261125.

Before submission, verify the exchange calendar and successfully read today's MD/JSON.
Only explicit missing files qualify. Network/auth failures, mismatched dates, malformed data,
or incomplete products are errors, not evidence of missing files.
Check repository Actions runs successfully: daily.yml must have no active requested/queued/waiting/pending/in_progress run.
Use a single request for a date; check request commit history before writing, and never
resubmit automatically after uncertain success. The bridge also rejects a repeated date
and same-day prior direct dispatch, and never dispatches on a rerun attempt.
The exchange-calendars XSHG calendar is pinned; unsupported calendar ranges fail closed.

The bridge uses its own GITHUB_TOKEN with contents:read and actions:write to invoke
daily.yml with ref=main, inputs.session=opt, inputs.date=request date.
It does not copy collection code or alter daily.yml or existing schedules.
A request commit or bridge success alone does not prove data collection succeeded.
Verify the bridge run, downstream daily.yml workflow_dispatch run and final MD/JSON internal date
and five-product completeness. 21:35 checks only; it never submits a request.
If unsupported or failed, retain the existing local retry backup and report the exact error.
