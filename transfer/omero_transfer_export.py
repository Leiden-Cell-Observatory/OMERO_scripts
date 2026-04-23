# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "omero-py>=5.21.0",
# ]
# ///
"""
omero_transfer_export.py

Bulk-export OMERO Projects, Datasets, Screens, and Plates as `omero transfer pack`
archives, one archive per object, named after the object. Works as both an
interactive marimo webapp and a headless CLI script.

Interactive:
    marimo edit omero_transfer_export.py
    marimo run  omero_transfer_export.py

CLI (note the `--` separator before notebook args):
    python omero_transfer_export.py -- \
        --server omero.example.org --username alice --password secret \
        --target-user bob --output ./out --resume

    python omero_transfer_export.py -- \
        --objects "Project:1,Dataset:5,Screen:3" --output ./out

Requires `omero-cli-transfer` installed in the same environment as `omero-py`.
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import re
    import subprocess
    import sys
    import time
    from pathlib import Path
    return Path, mo, re, subprocess, sys, time


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    _raw_args = mo.cli_args()
    if hasattr(_raw_args, "to_dict"):
        args = _raw_args.to_dict()
    else:
        args = dict(_raw_args)
    return args, is_script_mode


@app.cell
def _():
    def get_current_session():
        """Read last-used server/user from ~/.omero/sessions for form defaults."""
        try:
            from omero.util.sessions import SessionsStore
            store = SessionsStore()
            srv, usr, uuid, port = store.get_current()
            return srv, usr, uuid, int(port) if port else 4064
        except Exception:
            return None, None, None, None

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
def _():
    def connect(server, port, username=None, password=None, session_key=None):
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
        return conn
    return (connect,)


@app.cell
def _(mo):
    mo.md("# OMERO bulk transfer export")
    return


@app.cell
def _():
    def bool_arg(raw, default=False):
        """Coerce a cli_args value (string or bool) to a bool."""
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")
    return (bool_arg,)


@app.cell
def _(args, mo, sess_port, sess_server, sess_user):
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
    mo.vstack([
        mo.md("## Connection"),
        server_input,
        port_input,
        username_input,
        password_input,
    ])
    return password_input, port_input, server_input, username_input


@app.cell
def _(is_script_mode, mo):
    connect_button = mo.ui.run_button(label="Connect")
    connect_button if not is_script_mode else mo.md("_(script mode: auto-connects)_")
    return (connect_button,)


@app.cell
def _(
    connect,
    connect_button,
    is_script_mode,
    mo,
    password_input,
    port_input,
    sess_key,
    server_input,
    username_input,
):
    should_connect = is_script_mode or connect_button.value
    conn = None
    if should_connect:
        srv = server_input.value.strip()
        usr = username_input.value.strip()
        pwd = password_input.value
        prt = int(port_input.value or 4064)

        if sess_key and srv:
            conn = connect(srv, prt, session_key=sess_key)
        if conn is None and srv and usr and pwd:
            conn = connect(srv, prt, username=usr, password=pwd)

    if not should_connect:
        connect_status = mo.md("_Not connected yet._")
    elif conn is None:
        connect_status = mo.md("**❌ Could not connect — check server/credentials.**")
    else:
        connect_status = mo.md(
            f"**✅ Connected to `{server_input.value}` as `{conn.getUser().getName()}`**"
        )

    connect_status
    return (conn,)


@app.cell
def _(args, mo):
    default_mode = "By list" if args.get("objects") else "By user"
    source_mode = mo.ui.radio(
        options=["By user", "By list"],
        value=default_mode,
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

    mo.vstack([
        mo.md("## What to export"),
        source_mode,
        target_user_input,
        objects_input,
    ])
    return objects_input, source_mode, target_user_input


@app.cell
def _(re, slugify):
    def parse_object_list(text):
        items = []
        errors = []
        for token in re.split(r"[,\n]", text or ""):
            token = token.strip()
            if not token:
                continue
            m = re.fullmatch(
                r"(Project|Dataset|Screen|Plate):(\d+)", token, re.IGNORECASE
            )
            if m:
                items.append((m.group(1).capitalize(), int(m.group(2))))
            else:
                errors.append(token)
        return items, errors

    def discover_jobs(conn, source_mode_value, target_user, objects_text):
        """Return (jobs, warnings) where jobs = [(type, id, name)]."""
        jobs = []
        warnings = []
        if conn is None:
            return jobs, warnings

        conn.SERVICE_OPTS.setOmeroGroup("-1")

        if source_mode_value == "By user":
            target_user = (target_user or "").strip()
            if not target_user:
                return jobs, warnings
            exp = conn.getObject("Experimenter", attributes={"omeName": target_user})
            if exp is None:
                warnings.append(f"User '{target_user}' not found.")
                return jobs, warnings
            owner_opts = {"owner": exp.id}
            queries = [
                ("Project", owner_opts),
                ("Dataset", {**owner_opts, "orphaned": True}),
                ("Screen", owner_opts),
                ("Plate", {**owner_opts, "orphaned": True}),
            ]
            for obj_type, opts in queries:
                for o in conn.getObjects(obj_type, opts=opts):
                    jobs.append((obj_type, o.getId(), o.getName()))
        else:
            items, errors = parse_object_list(objects_text)
            for bad in errors:
                warnings.append(f"Ignoring malformed entry: '{bad}'")
            for obj_type, obj_id in items:
                try:
                    obj = conn.getObject(obj_type, obj_id)
                except Exception as exc:
                    warnings.append(f"{obj_type}:{obj_id} error: {exc}")
                    continue
                if obj is None:
                    warnings.append(f"{obj_type}:{obj_id} not found or not accessible.")
                    continue
                jobs.append((obj_type, int(obj_id), obj.getName()))
        return jobs, warnings

    def assign_output_names(jobs, extension):
        """Attach an output filename to each job, disambiguating duplicates."""
        used = set()
        out = []
        for obj_type, obj_id, name in jobs:
            slug = slugify(name)
            candidate = f"{slug}.{extension}"
            if candidate in used:
                candidate = f"{slug}_{obj_id}.{extension}"
            used.add(candidate)
            out.append((obj_type, obj_id, name, candidate))
        return out

    return assign_output_names, discover_jobs


@app.cell
def _(args, bool_arg, mo):
    output_input = mo.ui.text(
        label="Output directory",
        value=str(args.get("output", ".")),
        full_width=True,
    )
    zip_checkbox = mo.ui.checkbox(
        value=bool_arg(args.get("zip"), default=True),
        label="ZIP output (--zip)",
    )
    simple_checkbox = mo.ui.checkbox(
        value=bool_arg(args.get("simple"), default=False),
        label="Simple layout (--simple)",
    )
    binaries_none_checkbox = mo.ui.checkbox(
        value=bool_arg(args.get("binaries-none") or args.get("binaries_none"), default=False),
        label="Metadata only (--binaries none)",
    )
    resume_checkbox = mo.ui.checkbox(
        value=bool_arg(args.get("resume"), default=False),
        label="Resume (skip existing outputs)",
    )
    max_retries_input = mo.ui.number(
        label="Max retries per object",
        value=int(args.get("max-retries") or args.get("max_retries") or 5),
        start=1,
        stop=50,
    )
    mo.vstack([
        mo.md("## Options"),
        output_input,
        mo.hstack([zip_checkbox, simple_checkbox, binaries_none_checkbox, resume_checkbox]),
        max_retries_input,
    ])
    return (
        binaries_none_checkbox,
        max_retries_input,
        output_input,
        resume_checkbox,
        simple_checkbox,
        zip_checkbox,
    )


@app.cell
def _(
    assign_output_names,
    conn,
    discover_jobs,
    mo,
    objects_input,
    source_mode,
    target_user_input,
    zip_checkbox,
):
    jobs_raw, jobs_warnings = discover_jobs(
        conn,
        source_mode.value,
        target_user_input.value,
        objects_input.value,
    )
    ext = "zip" if zip_checkbox.value else "tar"
    jobs = assign_output_names(jobs_raw, ext)

    if jobs:
        table_rows = [
            {"Type": t, "ID": i, "Name": n, "Output": out}
            for t, i, n, out in jobs
        ]
        jobs_table = mo.ui.table(
            table_rows,
            selection="multi",
            page_size=25,
        )
    else:
        jobs_table = None

    warnings_md = (
        mo.md("\n".join(f"- ⚠️ {w}" for w in jobs_warnings))
        if jobs_warnings
        else mo.md("")
    )

    mo.vstack([
        mo.md(f"## Discovered jobs: {len(jobs)}"),
        warnings_md,
        jobs_table if jobs_table is not None else mo.md("_Nothing to export yet._"),
    ])
    return jobs, jobs_table


@app.cell
def _(is_script_mode, mo):
    run_button = mo.ui.run_button(label="Run export", kind="success")
    run_button if not is_script_mode else mo.md("_(script mode: auto-runs)_")
    return (run_button,)


@app.cell
def _(subprocess, time):
    def run_transfer_pack(
        obj_type,
        obj_id,
        output_path,
        server,
        port,
        username,
        password,
        zip_flag,
        simple,
        binaries_none,
        max_retries,
    ):
        """
        Run `omero transfer pack` with retry on failure.
        Deletes any partial output before retrying.
        Returns True on success, False after exhausting retries.
        """
        auth_args = ["-s", server, "-u", username, "-w", password]
        if port and int(port) != 4064:
            auth_args += ["-p", str(port)]

        cmd = ["omero", "transfer"] + auth_args + ["pack"]
        if zip_flag:
            cmd.append("--zip")
        if simple:
            cmd.append("--simple")
        if binaries_none:
            cmd += ["--binaries", "none"]
        cmd += [f"{obj_type}:{obj_id}", str(output_path)]

        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                print(f"  Retry {attempt}/{max_retries} after 5s...")
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except OSError:
                        pass
                time.sleep(5)
            result = subprocess.run(cmd)
            if result.returncode == 0:
                return True
            print(f"  Attempt {attempt} failed (exit {result.returncode}).")
        return False
    return (run_transfer_pack,)


@app.cell
def _(
    Path,
    binaries_none_checkbox,
    conn,
    is_script_mode,
    jobs,
    jobs_table,
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
    sys,
    username_input,
    zip_checkbox,
):
    should_run = is_script_mode or run_button.value
    log_lines = []

    if should_run and conn is not None and jobs:
        if is_script_mode:
            selected = jobs
        else:
            picked = jobs_table.value if jobs_table is not None else None
            if picked:
                keys = {(row["Type"], row["ID"]) for row in picked}
                selected = [j for j in jobs if (j[0], j[1]) in keys]
            else:
                selected = jobs  # default: all

        output_dir = Path(output_input.value or ".").expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        skipped = 0
        failed = 0

        for idx, (obj_type, obj_id, name, out_filename) in enumerate(selected, 1):
            out_path = output_dir / out_filename
            header = f"({idx}/{len(selected)}) {obj_type}:{obj_id}  [{name}]  →  {out_path}"
            print(header)
            log_lines.append(header)

            if resume_checkbox.value and out_path.exists():
                msg = "  ↷ already exists, skipping."
                print(msg)
                log_lines.append(msg)
                skipped += 1
                continue

            ok = run_transfer_pack(
                obj_type,
                obj_id,
                out_path,
                server_input.value.strip(),
                port_input.value,
                username_input.value.strip(),
                password_input.value,
                zip_checkbox.value,
                simple_checkbox.value,
                binaries_none_checkbox.value,
                int(max_retries_input.value),
            )
            if ok:
                exported += 1
                log_lines.append("  ✓ exported")
            else:
                failed += 1
                log_lines.append("  ✗ FAILED after retries")

        summary = (
            f"**Done:** {exported} exported, {skipped} skipped, {failed} failed "
            f"({exported + skipped}/{len(selected)} OK)."
        )
        print(summary.replace("**", ""))
        log_lines.append(summary)

        if is_script_mode and failed > 0:
            sys.exit(1)

    if not should_run:
        status_md = mo.md("_Click **Run export** to start._")
    elif conn is None:
        status_md = mo.md("**Not connected — cannot run.**")
    elif not jobs:
        status_md = mo.md("**No jobs to export.**")
    else:
        status_md = mo.md("\n\n".join(log_lines))

    status_md
    return


if __name__ == "__main__":
    app.run()
