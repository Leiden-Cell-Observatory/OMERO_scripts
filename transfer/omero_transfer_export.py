# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "marimo",
#     "omero-py>=5.21.0",
#     "omero-cli-transfer",
#     "typer>=0.12",
#     "rich",
#     "zeroc-ice",
# ]
#
# # Glencoe Software publishes prebuilt zeroc-ice wheels per OS / Python version.
# # These wheels are pinned to cp310 to match requires-python above. If you bump
# # the Python version, update each URL to the matching cp3XX wheel from the
# # same Glencoe release (see https://github.com/glencoesoftware).
# [tool.uv.sources]
# zeroc-ice = [
#     { url = "https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases/download/20240202/zeroc_ice-3.6.5-cp310-cp310-manylinux_2_28_x86_64.whl", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" },
#     { url = "https://github.com/glencoesoftware/zeroc-ice-py-macos-universal2/releases/download/20240131/zeroc_ice-3.6.5-cp310-cp310-macosx_11_0_universal2.whl", marker = "sys_platform == 'darwin'" },
#     { url = "https://github.com/glencoesoftware/zeroc-ice-py-win-amd64/releases/download/20240325/zeroc_ice-3.6.5-cp310-cp310-win_amd64.whl", marker = "sys_platform == 'win32'" },
# ]
# ///
"""
omero_transfer_export.py

Bulk-export OMERO Projects, Datasets, Screens, and Plates as `omero transfer pack`
archives, one archive per object, named after the object. Works as:

  1. An interactive marimo webapp:   marimo edit omero_transfer_export.py
  2. A guided interactive CLI:       python omero_transfer_export.py
     (prompts for every missing value; pre-fills from ~/.omero/sessions)
  3. A non-interactive CLI:          python omero_transfer_export.py -- \
                                        --server ... --target-user bob --output ./out

By default each object becomes a folder of files (human-readable layout via
`--simple`). Pass `--format zip` or `--format tar` for a single-archive output.
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import re
    import shutil
    import subprocess
    import sys
    import time
    from dataclasses import dataclass
    from pathlib import Path

    SUPPORTED_TYPES = ("Project", "Dataset", "Screen", "Plate")
    DEFAULT_PORT = 4064
    RETRY_DELAY = 5
    return (
        DEFAULT_PORT,
        Path,
        RETRY_DELAY,
        SUPPORTED_TYPES,
        dataclass,
        mo,
        re,
        shutil,
        subprocess,
        sys,
        time,
    )


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    _raw = mo.cli_args()
    args = _raw.to_dict() if hasattr(_raw, "to_dict") else dict(_raw)
    return args, is_script_mode


@app.cell
def _(DEFAULT_PORT):
    def get_current_session():
        """Read last-used server/user from ~/.omero/sessions for form defaults."""
        try:
            from omero.util.sessions import SessionsStore
            srv, usr, uuid, port = SessionsStore().get_current()
            return srv, usr, uuid, int(port) if port else DEFAULT_PORT
        except Exception:
            return None, None, None, DEFAULT_PORT

    sess_server, sess_user, sess_key, sess_port = get_current_session()
    return sess_key, sess_port, sess_server, sess_user


@app.cell
def _(re):
    def slugify(name):
        name = (name or "").strip().replace(" ", "_")
        name = re.sub(r"[^\w\-.]", "_", name)
        return name or "unnamed"
    return (slugify,)


@app.cell
def _(DEFAULT_PORT):
    def connect(server, port=DEFAULT_PORT, username=None, password=None,
                session_key=None, group=None):
        from omero.gateway import BlitzGateway
        if session_key:
            conn = BlitzGateway(host=server, port=port, secure=True)
            conn.connect(sUuid=session_key)
        else:
            conn = BlitzGateway(username, password, host=server, port=port, secure=True)
            conn.connect()
        if not conn.isConnected():
            return None
        conn.c.enableKeepAlive(60)
        if group:
            group_id = resolve_group_id(conn, group)
            if group_id is not None:
                conn.setGroupForSession(group_id)
        return conn

    def resolve_group_id(conn, group):
        """Accept either a group ID (int/str-digit) or a group name; return int ID or None."""
        if group is None or group == "":
            return None
        s = str(group).strip()
        if s.isdigit():
            return int(s)
        g = conn.getObject("ExperimenterGroup", attributes={"name": s})
        return g.getId() if g is not None else None

    def list_groups(conn):
        """Return [(id, name)] for groups the current user is a member of."""
        exp = conn.getUser()
        groups = []
        for g in exp.copyGroupExperimenterMap():
            grp = g.parent
            gid = grp.getId().val if hasattr(grp.getId(), "val") else grp.getId()
            gname = grp.getName().val if hasattr(grp.getName(), "val") else grp.getName()
            if gname == "user":  # skip the implicit "user" group
                continue
            groups.append((int(gid), str(gname)))
        return groups
    return connect, list_groups, resolve_group_id


@app.cell
def _(SUPPORTED_TYPES, re):
    def parse_object_list(text):
        """Return ([(Type, id)], [bad_tokens])."""
        pattern = re.compile(
            rf"({'|'.join(SUPPORTED_TYPES)}):(\d+)", re.IGNORECASE
        )
        items, errors = [], []
        for token in re.split(r"[,\n]", text or ""):
            token = token.strip()
            if not token:
                continue
            m = pattern.fullmatch(token)
            if m:
                items.append((m.group(1).capitalize(), int(m.group(2))))
            else:
                errors.append(token)
        return items, errors
    return (parse_object_list,)


@app.cell
def _(SUPPORTED_TYPES, parse_object_list):
    def discover_jobs(conn, source_mode, target_user=None, objects_text=None):
        """
        Return ([(type, id, name)], [warnings]).
        source_mode: "user" enumerates the user's Projects/orphan-Datasets/Screens/orphan-Plates.
                     "list"  resolves a literal Type:ID list.
        """
        jobs, warnings = [], []
        if conn is None:
            return jobs, warnings
        conn.SERVICE_OPTS.setOmeroGroup("-1")

        if source_mode == "user":
            user = (target_user or "").strip()
            if not user:
                return jobs, warnings
            exp = conn.getObject("Experimenter", attributes={"omeName": user})
            if exp is None:
                warnings.append(f"User '{user}' not found.")
                return jobs, warnings
            owner = {"owner": exp.id}
            for obj_type, opts in (
                ("Project", owner),
                ("Dataset", {**owner, "orphaned": True}),
                ("Screen", owner),
                ("Plate", {**owner, "orphaned": True}),
            ):
                for o in conn.getObjects(obj_type, opts=opts):
                    jobs.append((obj_type, o.getId(), o.getName()))
        else:
            items, bad = parse_object_list(objects_text)
            for token in bad:
                warnings.append(
                    f"Ignoring '{token}' (expected {'|'.join(SUPPORTED_TYPES)}:ID)"
                )
            for obj_type, obj_id in items:
                obj = conn.getObject(obj_type, obj_id)
                if obj is None:
                    warnings.append(f"{obj_type}:{obj_id} not found or not accessible.")
                    continue
                jobs.append((obj_type, obj_id, obj.getName()))
        return jobs, warnings
    return (discover_jobs,)


@app.cell
def _(slugify):
    OUTPUT_FORMATS = ("folder", "zip", "tar")

    def assign_output_names(jobs, output_format):
        """
        Attach an output name to each job, disambiguating duplicates with _<id>.
        output_format: "folder" → <slug> (no extension, omero writes a directory),
                       "zip"    → <slug>.zip,
                       "tar"    → <slug>.tar.
        """
        suffix = "" if output_format == "folder" else f".{output_format}"
        used, out = set(), []
        for obj_type, obj_id, name in jobs:
            slug = slugify(name)
            candidate = f"{slug}{suffix}"
            if candidate in used:
                candidate = f"{slug}_{obj_id}{suffix}"
            used.add(candidate)
            out.append((obj_type, obj_id, name, candidate))
        return out
    return OUTPUT_FORMATS, assign_output_names


@app.cell
def _(DEFAULT_PORT, RETRY_DELAY, dataclass, shutil, subprocess, time):
    @dataclass
    class TransferOptions:
        simple: bool = True  # human-readable project/dataset/images layout
        binaries_none: bool = False
        max_retries: int = 5

    @dataclass
    class OmeroAuth:
        server: str
        username: str
        password: str
        port: int = DEFAULT_PORT
        group: str = ""  # group name or ID; empty = user's default group

    def cleanup_path(path):
        """Remove a file or directory if it exists. Used before a retry or first attempt."""
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    def run_transfer_pack(obj_type, obj_id, output_path, auth, opts, log=print):
        """
        Run `omero transfer pack`, retrying up to opts.max_retries times on failure.
        Output format is chosen by the extension of `output_path`:
          .zip/.tar → archive; no extension → a folder of files.
        Any pre-existing partial output is removed before the first attempt.
        """
        cleanup_path(output_path)

        cmd = ["omero", "transfer", "-s", auth.server, "-u", auth.username, "-w", auth.password]
        if auth.port and int(auth.port) != DEFAULT_PORT:
            cmd += ["-p", str(auth.port)]
        if auth.group:
            cmd += ["-g", str(auth.group)]
        cmd += ["pack"]
        if opts.simple:
            cmd.append("--simple")
        if opts.binaries_none:
            cmd += ["--binaries", "none"]
        cmd += [f"{obj_type}:{obj_id}", str(output_path)]

        for attempt in range(1, opts.max_retries + 1):
            if attempt > 1:
                log(f"  Retry {attempt}/{opts.max_retries} in {RETRY_DELAY}s...")
                cleanup_path(output_path)
                time.sleep(RETRY_DELAY)
            if subprocess.run(cmd).returncode == 0:
                return True
            log(f"  Attempt {attempt} failed.")
        return False
    return OmeroAuth, TransferOptions, run_transfer_pack


# ---------------------------------------------------------------------------
# Script-mode entry point: guided interactive CLI via typer/rich
# ---------------------------------------------------------------------------


@app.cell
def _(
    DEFAULT_PORT,
    OUTPUT_FORMATS,
    OmeroAuth,
    Path,
    SUPPORTED_TYPES,
    TransferOptions,
    args,
    assign_output_names,
    connect,
    discover_jobs,
    is_script_mode,
    list_groups,
    resolve_group_id,
    run_transfer_pack,
    sess_key,
    sess_port,
    sess_server,
    sess_user,
    sys,
):
    def _script_main():
        import typer
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.rule("[bold blue]OMERO bulk transfer export[/bold blue]")

        server = args.get("server") or typer.prompt("OMERO server", default=sess_server or "")
        port = int(args.get("port") or sess_port or DEFAULT_PORT)
        username = args.get("username") or typer.prompt("Username", default=sess_user or "")
        password = args.get("password") or typer.prompt("Password", hide_input=True)

        console.print(f"[dim]Connecting to {server}...[/dim]")
        conn = None
        if sess_key:
            conn = connect(server, port, session_key=sess_key)
        if conn is None:
            conn = connect(server, port, username=username, password=password)
        if conn is None:
            console.print("[red]Could not connect. Check credentials/server.[/red]")
            raise typer.Exit(1)
        console.print(f"[green]Connected as {conn.getUser().getName()}[/green]")

        group = args.get("group") or ""
        if not group:
            available = list_groups(conn)
            if available:
                group_table = Table(title="Your groups")
                group_table.add_column("ID", justify="right")
                group_table.add_column("Name")
                for gid, gname in available:
                    group_table.add_row(str(gid), gname)
                console.print(group_table)
            group = typer.prompt(
                "OMERO group (name or ID, blank = default group)", default=""
            ).strip()
        if group:
            gid = resolve_group_id(conn, group)
            if gid is None:
                console.print(f"[red]Group '{group}' not found.[/red]")
                raise typer.Exit(1)
            conn.setGroupForSession(gid)
            console.print(f"[green]Switched to group {group} (id={gid})[/green]")

        try:
            if args.get("objects"):
                source_mode = "list"
                objects_text = args["objects"]
                target_user = None
            elif args.get("target-user") or args.get("target_user"):
                source_mode = "user"
                target_user = args.get("target-user") or args.get("target_user")
                objects_text = None
            else:
                choice = typer.prompt(
                    "Source: [1] all objects of a user  [2] explicit Type:ID list",
                    default="1",
                )
                if choice.strip().startswith("2"):
                    source_mode = "list"
                    objects_text = typer.prompt(
                        f"Object list ({'|'.join(SUPPORTED_TYPES)}:ID, comma-separated)"
                    )
                    target_user = None
                else:
                    source_mode = "user"
                    target_user = typer.prompt(
                        "Target user", default=username
                    )
                    objects_text = None

            jobs_raw, warnings = discover_jobs(conn, source_mode, target_user, objects_text)
            for w in warnings:
                console.print(f"[yellow]⚠ {w}[/yellow]")
            if not jobs_raw:
                console.print("[yellow]No objects to export.[/yellow]")
                raise typer.Exit(0)

            output_dir = Path(
                args.get("output") or typer.prompt("Output directory", default=".")
            ).expanduser()

            output_format = args.get("format")
            if output_format not in OUTPUT_FORMATS:
                output_format = typer.prompt(
                    f"Output format [{'/'.join(OUTPUT_FORMATS)}]", default="folder"
                ).strip().lower()
                if output_format not in OUTPUT_FORMATS:
                    console.print(f"[red]Unknown format '{output_format}'.[/red]")
                    raise typer.Exit(2)

            opts = TransferOptions(
                simple=bool(args.get("simple", True)) if "simple" in args else typer.confirm(
                    "Use human-readable layout (--simple)?", default=True
                ),
                binaries_none=bool(args.get("binaries-none") or args.get("binaries_none"))
                if ("binaries-none" in args or "binaries_none" in args) else False,
                max_retries=int(args.get("max-retries") or args.get("max_retries") or 5),
            )
            resume = (
                bool(args.get("resume", False)) if "resume" in args
                else typer.confirm("Resume (skip existing outputs)?", default=True)
            )
        finally:
            pass  # keep conn open for any name lookups, closed below

        jobs = assign_output_names(jobs_raw, output_format)

        table = Table(title=f"{len(jobs)} object(s) to export")
        table.add_column("#")
        table.add_column("Type")
        table.add_column("ID", justify="right")
        table.add_column("Name")
        table.add_column("Output")
        for i, (t, oid, name, out) in enumerate(jobs, 1):
            table.add_row(str(i), t, str(oid), name, out)
        console.print(table)

        if not typer.confirm(f"Export all {len(jobs)} to {output_dir}?", default=True):
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

        conn.close()

        output_dir.mkdir(parents=True, exist_ok=True)
        auth = OmeroAuth(
            server=server, port=port, username=username, password=password, group=group
        )
        exported = skipped = failed = 0
        for idx, (t, oid, name, out_filename) in enumerate(jobs, 1):
            out_path = output_dir / out_filename
            console.print(
                f"\n[bold]({idx}/{len(jobs)})[/bold] {t}:{oid} [dim]{name}[/dim] → {out_path}"
            )
            if resume and out_path.exists():
                console.print("  [dim]already exists, skipping.[/dim]")
                skipped += 1
                continue
            if run_transfer_pack(t, oid, out_path, auth, opts, log=console.print):
                exported += 1
                console.print(f"  [green]✓ {out_path}[/green]")
            else:
                failed += 1
                console.print(f"  [red]✗ failed[/red]")

        console.rule()
        console.print(
            f"[bold]Done:[/bold] {exported} exported, {skipped} skipped, {failed} failed."
        )
        if failed:
            sys.exit(1)

    if is_script_mode:
        _script_main()
    return


# ---------------------------------------------------------------------------
# Webapp cells (inert in script mode because their run-buttons never trigger)
# ---------------------------------------------------------------------------


@app.cell
def _(is_script_mode, mo):
    mo.md("# OMERO bulk transfer export") if not is_script_mode else None
    return


@app.cell
def _(args, is_script_mode, mo, sess_port, sess_server, sess_user):
    server_input = mo.ui.text(
        label="OMERO server",
        value=str(args.get("server", sess_server or "")),
        full_width=True,
    )
    port_input = mo.ui.number(
        label="Port",
        value=int(args.get("port", sess_port or 4064)),
        start=1,
        stop=65535,
    )
    username_input = mo.ui.text(
        label="Username",
        value=str(args.get("username", sess_user or "")),
        full_width=True,
    )
    password_input = mo.ui.text(
        label="Password",
        value=str(args.get("password", "")),
        kind="password",
        full_width=True,
    )
    group_input = mo.ui.text(
        label="Group (name or ID, blank = default)",
        value=str(args.get("group", "")),
        full_width=True,
    )
    connection_form = mo.vstack([
        mo.md("## Connection"),
        server_input,
        port_input,
        username_input,
        password_input,
        group_input,
    ])
    connection_form if not is_script_mode else None
    return group_input, password_input, port_input, server_input, username_input


@app.cell
def _(is_script_mode, mo):
    connect_button = mo.ui.run_button(label="Connect")
    connect_button if not is_script_mode else None
    return (connect_button,)


@app.cell
def _(
    connect,
    connect_button,
    group_input,
    is_script_mode,
    mo,
    password_input,
    port_input,
    sess_key,
    server_input,
    username_input,
):
    web_conn = None
    if not is_script_mode and connect_button.value:
        srv, usr, pwd, prt, grp = (
            server_input.value.strip(),
            username_input.value.strip(),
            password_input.value,
            int(port_input.value or 4064),
            group_input.value.strip(),
        )
        if sess_key and srv:
            web_conn = connect(srv, prt, session_key=sess_key, group=grp)
        if web_conn is None and srv and usr and pwd:
            web_conn = connect(srv, prt, username=usr, password=pwd, group=grp)

    if is_script_mode:
        connect_status = None
    elif not connect_button.value:
        connect_status = mo.md("_Not connected yet._")
    elif web_conn is None:
        connect_status = mo.md("**❌ Could not connect — check server/credentials.**")
    else:
        current_group = web_conn.getGroupFromContext().getName()
        connect_status = mo.md(
            f"**✅ Connected to `{server_input.value}` as "
            f"`{web_conn.getUser().getName()}` (group: `{current_group}`)**"
        )
    connect_status
    return (web_conn,)


@app.cell
def _(args, is_script_mode, mo):
    source_mode_ui = mo.ui.radio(
        options=["By user", "By list"],
        value="By list" if args.get("objects") else "By user",
        label="Source",
    )
    target_user_input = mo.ui.text(
        label="Target user (OMERO username)",
        value=str(args.get("target-user", args.get("target_user", ""))),
        full_width=True,
    )
    objects_input = mo.ui.text_area(
        label="Object list (Type:ID, comma or newline separated)",
        placeholder="Project:1, Dataset:5, Screen:3, Plate:16",
        value=str(args.get("objects", "")),
        full_width=True,
        rows=4,
    )
    source_form = mo.vstack([
        mo.md("## What to export"),
        source_mode_ui,
        target_user_input,
        objects_input,
    ])
    source_form if not is_script_mode else None
    return objects_input, source_mode_ui, target_user_input


@app.cell
def _(is_script_mode, mo):
    discover_button = mo.ui.run_button(label="Discover objects")
    discover_button if not is_script_mode else None
    return (discover_button,)


@app.cell
def _(
    discover_button,
    discover_jobs,
    is_script_mode,
    mo,
    objects_input,
    source_mode_ui,
    target_user_input,
    web_conn,
):
    jobs_raw_web, jobs_warnings = [], []
    if not is_script_mode and discover_button.value and web_conn is not None:
        mode_key = "list" if source_mode_ui.value == "By list" else "user"
        jobs_raw_web, jobs_warnings = discover_jobs(
            web_conn, mode_key, target_user_input.value, objects_input.value
        )

    if is_script_mode:
        jobs_panel = None
    elif not discover_button.value:
        jobs_panel = mo.md("_Click **Discover objects** to enumerate._")
    elif web_conn is None:
        jobs_panel = mo.md("**Not connected — connect first.**")
    elif not jobs_raw_web:
        jobs_panel = mo.vstack([
            mo.md("_No objects found._"),
            *[mo.md(f"- ⚠ {w}") for w in jobs_warnings],
        ])
    else:
        jobs_panel = mo.vstack([
            mo.md(f"**{len(jobs_raw_web)} object(s) discovered**"),
            *[mo.md(f"- ⚠ {w}") for w in jobs_warnings],
            mo.ui.table(
                [
                    {"Type": t, "ID": i, "Name": n}
                    for t, i, n in jobs_raw_web
                ],
                selection="multi",
                page_size=25,
            ),
        ])
    jobs_panel
    return (jobs_raw_web,)


@app.cell
def _(OUTPUT_FORMATS, args, is_script_mode, mo):
    output_input = mo.ui.text(
        label="Output directory",
        value=str(args.get("output", ".")),
        full_width=True,
    )
    default_format = args.get("format") if args.get("format") in OUTPUT_FORMATS else "folder"
    format_radio = mo.ui.radio(
        options=list(OUTPUT_FORMATS),
        value=default_format,
        label="Output format (folder = raw files, zip/tar = archive)",
    )
    simple_checkbox = mo.ui.checkbox(
        value=True,
        label="Human-readable layout (--simple)",
    )
    binaries_none_checkbox = mo.ui.checkbox(value=False, label="Metadata only (--binaries none)")
    resume_checkbox = mo.ui.checkbox(value=False, label="Resume (skip existing)")
    max_retries_input = mo.ui.number(label="Max retries", value=5, start=1, stop=50)
    options_form = mo.vstack([
        mo.md("## Options"),
        output_input,
        format_radio,
        mo.hstack([simple_checkbox, binaries_none_checkbox, resume_checkbox]),
        max_retries_input,
    ])
    options_form if not is_script_mode else None
    return (
        binaries_none_checkbox,
        format_radio,
        max_retries_input,
        output_input,
        resume_checkbox,
        simple_checkbox,
    )


@app.cell
def _(is_script_mode, mo):
    run_button = mo.ui.run_button(label="Run export")
    run_button if not is_script_mode else None
    return (run_button,)


@app.cell
def _(
    OmeroAuth,
    Path,
    TransferOptions,
    assign_output_names,
    binaries_none_checkbox,
    format_radio,
    group_input,
    is_script_mode,
    jobs_raw_web,
    max_retries_input,
    mo,
    output_input,
    password_input,
    port_input,
    resume_checkbox,
    run_button,
    run_transfer_pack,
    server_input,
    simple_checkbox,
    username_input,
    web_conn,
):
    log_lines = []
    if not is_script_mode and run_button.value and web_conn is not None and jobs_raw_web:
        output_dir = Path(output_input.value or ".").expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        auth = OmeroAuth(
            server=server_input.value.strip(),
            port=int(port_input.value or 4064),
            username=username_input.value.strip(),
            password=password_input.value,
            group=group_input.value.strip(),
        )
        opts = TransferOptions(
            simple=simple_checkbox.value,
            binaries_none=binaries_none_checkbox.value,
            max_retries=int(max_retries_input.value),
        )
        jobs = assign_output_names(jobs_raw_web, format_radio.value)

        exported = skipped = failed = 0
        for idx, (t, oid, name, out_filename) in enumerate(jobs, 1):
            out_path = output_dir / out_filename
            header = f"({idx}/{len(jobs)}) {t}:{oid} [{name}] → {out_path}"
            log_lines.append(header)
            if resume_checkbox.value and out_path.exists():
                log_lines.append("  ↷ already exists, skipping.")
                skipped += 1
                continue
            if run_transfer_pack(t, oid, out_path, auth, opts, log=log_lines.append):
                exported += 1
                log_lines.append("  ✓ exported")
            else:
                failed += 1
                log_lines.append("  ✗ FAILED after retries")
        log_lines.append(
            f"**Done:** {exported} exported, {skipped} skipped, {failed} failed."
        )

    if is_script_mode:
        export_panel = None
    elif not run_button.value:
        export_panel = mo.md("_Configure options above, then **Run export**._")
    elif web_conn is None:
        export_panel = mo.md("**Not connected.**")
    elif not jobs_raw_web:
        export_panel = mo.md("**No jobs — click Discover first.**")
    else:
        export_panel = mo.md("\n\n".join(log_lines))
    export_panel
    return


if __name__ == "__main__":
    app.run()
