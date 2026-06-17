Parse proposed JIRAs from a spike doc and file them via the Jira API

You are filing JIRA sub-tickets for a Lightspeed Core feature.

The user will provide either a spike doc path or tell you which feature's
JIRAs to file.  They will also provide the feature ticket number.

Run `sh dev-tools/file-jiras.sh --help` to see the full usage.

## Credentials

Jira credentials are managed by `dev-tools/jira-common.sh`.  If
`~/.config/jira/credentials.json` doesn't exist, the script creates it
with FIXMEs and exits — the user must fill in their credentials before
re-running.  API tokens can be created at
https://id.atlassian.com/manage-profile/security/api-tokens

## Spike doc shape the parser expects

`file-jiras.sh` parses the spike doc's `## Proposed JIRAs` section into
Epic-grouped tickets:

```markdown
## Proposed JIRAs

### Epic: <epic-name>

<epic prose: Goals, optional Scope, Success criteria>

#### LCORE-???? <child title>     ← H4, child of the Epic above
<child body>

#### LCORE-???? <another child>
<child body>

### Epic: <another epic-name>     ← optional second Epic
...
```

Each `### Epic: <name>` becomes a filed Epic; each `#### LCORE-????`
under it becomes a child of that Epic. Children carry a
`<!-- parent_epic_file: <stub> -->` metadata comment in their parsed
files; the script uses this to route each child to its parent Epic's
filed key at filing time.

**Backward-compat**: spike docs without `### Epic:` boundaries (flat
`### LCORE-...` H3 stubs directly under `## Proposed JIRAs`) still
parse — they get a single auto-generated Epic derived from the spike
doc's parent directory name.

**Already-filed keys**: if a heading reads `### LCORE-1569: <title>`
(or `#### LCORE-1569: ...`), the parser preserves the real key in the
output file. At filing time, the script sends a PUT (update) instead
of a POST (create), useful for re-syncing previously-filed tickets
with updated descriptions.

**Incidental tickets** (under `## Proposed incidental JIRAs`) file
under the feature ticket directly, not under any Epic.

## Process

1. Run `dev-tools/file-jiras.sh --spike-doc <path> --feature-ticket <key> --parse-only`
   to parse the spike doc into ticket files and exit. `--parse-only` skips
   the interactive filing loop and the credentials check, so it works
   even on machines without Jira credentials configured (CI, agent
   inspection, pre-commit hooks).

2. Read every file in the output directory (default: `docs/design/<feature>/jiras/`).
   For each, verify:
   - Content matches the corresponding section in the spike doc (no truncation,
     no extra content swallowed from subsequent sections).
   - File size is reasonable (a single JIRA should be under ~3KB; if any file
     is much larger, the parser likely grabbed too much).
   - The `<!-- type: ... -->` metadata is correct (Epic/Story/Task).
   - For children: `<!-- parent_epic_file: <stub> -->` points at an
     existing Epic file in the same directory.

3. Watch the parser's stderr for `[LINT-WARNING]` lines (mixed shape,
   empty Epics, duplicate titles). `[LINT-ERROR]` causes the parser
   to exit non-zero — fix the spike doc and re-run.

4. Report any issues to the user.  If all files look correct, tell the user
   to run the script interactively — provide the full command including `cd`
   to the repository root:
   `cd <repo-path> && sh dev-tools/file-jiras.sh --spike-doc <path> --feature-ticket <key>`

## Filing order

The script files Epics first, then their children. With multi-Epic
spike docs, this means Epic A is filed → its children land under Epic
A → Epic B is filed → its children land under Epic B. Children whose
parent Epic hasn't been filed yet error out clearly; file the Epic
first and retry.

Incidental tickets file last (under FEATURE_TICKET, no Epic parent).
