"""
omero_zarr_export.py

Export plates from OMERO as OME-Zarr files, optionally discovering all plates
within a screen. Wraps the omero-cli-zarr plugin (`omero zarr export`) and
handles naming, folder structure, optional zipping, and resume.

Usage examples:
  python omero_zarr_export.py
  python omero_zarr_export.py --object Screen:5 --output ./zarr_exports
  python omero_zarr_export.py --object Screen:5 --output ./zarr_exports --resume
  python omero_zarr_export.py --object Plate:16 --server omero.example.org --zip

Requirements:
  pip install typer rich omero-py
  omero-cli-zarr must be installed in the same environment
"""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer
from omero.gateway import BlitzGateway
from rich.console import Console

console = Console()
app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_current_session():
    """
    Read the last-used server and username from the omero CLI session store
    (~/.omero/sessions). Used only to pre-fill interactive prompts.
    Returns (server, username, session_key, port) or (None, None, None, None).
    """
    try:
        from omero.util.sessions import SessionsStore
        store = SessionsStore()
        srv, usr, uuid, port = store.get_current()
        return srv, usr, uuid, int(port) if port else 4064
    except Exception:
        return None, None, None, None


def slugify(name: str) -> str:
    """Convert an arbitrary OMERO name to a safe folder name."""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^\w\-.]", "_", name)
    return name or "unnamed"


def connect(server: str, port: int, username: str = None,
            password: str = None, session_key: str = None) -> Optional[BlitzGateway]:
    """
    Connect to OMERO via session key or username/password.
    Returns None on failure so the caller can decide to retry.
    """
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


def get_plates_for_screen(conn: BlitzGateway, screen_id: int):
    """Return (screen_name, [(plate_id, plate_name), ...]) for all plates in a screen."""
    screen = conn.getObject("Screen", screen_id)
    if screen is None:
        console.print(f"[red]Screen:{screen_id} not found or not accessible.[/red]")
        raise typer.Exit(1)
    plates = [(pl.getId(), pl.getName()) for pl in screen.listChildren()]
    return screen.getName(), plates


def get_plate_name(conn: BlitzGateway, plate_id: int) -> str:
    """Return the name of a single plate."""
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        console.print(f"[red]Plate:{plate_id} not found or not accessible.[/red]")
        raise typer.Exit(1)
    return plate.getName()


def run_zarr_export(
    plate_id: int,
    output_dir: Path,
    server: str,
    port: int,
    username: str,
    password: str,
    tile_width: Optional[int],
    tile_height: Optional[int],
) -> bool:
    """
    Call `omero zarr export Plate:ID` as a subprocess using credentials.
    Always uses username/password to avoid stale session key issues.
    Output is streamed directly to the terminal.
    Returns True on success, False on failure.
    """
    auth_args = ["-s", server, "-u", username, "-w", password]

    # Only pass port if non-default to avoid "port specified twice" errors
    if port != 4064:
        auth_args += ["-p", str(port)]

    cmd = (
        ["omero", "zarr"]
        + auth_args
        + ["--output", str(output_dir), "export", f"Plate:{plate_id}"]
    )
    if tile_width:
        cmd += ["--tile_width", str(tile_width)]
    if tile_height:
        cmd += ["--tile_height", str(tile_height)]

    # Run without capturing output so omero zarr progress is visible in terminal
    result = subprocess.run(cmd)
    if result.returncode != 0:
        console.print(f"[red]Export failed for Plate:{plate_id}[/red]")
        return False
    return True


def rename_zarr(output_dir: Path, plate_id: int, plate_name: str) -> Path:
    """
    Rename <output_dir>/<plate_id>.ome.zarr → <output_dir>/<safe_plate_name>.ome.zarr.
    If the target name already exists (duplicate plate names are possible in OMERO),
    appends _2, _3, etc. Falls back to ID-based name if the exported folder is missing.
    """
    id_path = output_dir / f"{plate_id}.ome.zarr"
    safe_name = slugify(plate_name)

    if not id_path.exists():
        console.print(f"[yellow]Warning: expected {id_path} not found after export.[/yellow]")
        return id_path

    # Find a unique name, appending a counter for duplicates
    candidate = output_dir / f"{safe_name}.ome.zarr"
    counter = 2
    while candidate.exists():
        candidate = output_dir / f"{safe_name}_{counter}.ome.zarr"
        counter += 1

    if counter > 2:
        console.print(f"  [yellow]Duplicate name '{safe_name}', using '{candidate.name}'[/yellow]")

    id_path.rename(candidate)
    return candidate


def zip_zarr(zarr_path: Path) -> Path:
    """Zip a zarr folder and remove the original. Returns path to the zip file."""
    archive = shutil.make_archive(str(zarr_path), "zip", zarr_path.parent, zarr_path.name)
    shutil.rmtree(zarr_path)
    return Path(archive)


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------

@app.command()
def main(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="OMERO server hostname"),
    port: int = typer.Option(4064, "--port", "-p", help="OMERO server port"),
    username: Optional[str] = typer.Option(None, "--username", "-u", help="OMERO username"),
    password: Optional[str] = typer.Option(None, "--password", "-w", help="OMERO password"),
    object_id: Optional[str] = typer.Option(
        None, "--object", "-o",
        help="Object to export, e.g. Screen:5 or Plate:16"
    ),
    output: Path = typer.Option(Path("."), "--output", help="Base output directory"),
    resume: bool = typer.Option(
        False, "--resume",
        help="Skip plates already successfully exported; retry partial exports"
    ),
    zip_output: bool = typer.Option(False, "--zip", help="Zip each zarr folder after export"),
    tile_width: Optional[int] = typer.Option(None, "--tile-width", help="Zarr chunk tile width"),
    tile_height: Optional[int] = typer.Option(None, "--tile-height", help="Zarr chunk tile height"),
):
    """Export OMERO plates (or all plates in a screen) as OME-Zarr files."""

    console.rule("[bold blue]OMERO Zarr Export[/bold blue]")

    # Read session store only to pre-fill server and username defaults
    sess_server, sess_user, session_key, sess_port = get_current_session()

    # Prompt for any missing credentials, using session store values as defaults
    if not server:
        server = typer.prompt("OMERO server", default=sess_server or "")
    if not username:
        username = typer.prompt("Username", default=sess_user or "")
    if not password:
        password = typer.prompt("Password", hide_input=True)

    if not object_id:
        object_id = typer.prompt("Object to export (e.g. Screen:5 or Plate:16)")

    # Parse object type and ID
    match = re.fullmatch(r"(Screen|Plate):(\d+)", object_id.strip(), re.IGNORECASE)
    if not match:
        console.print("[red]Object must be in the format Screen:ID or Plate:ID[/red]")
        raise typer.Exit(1)
    obj_type = match.group(1).capitalize()
    obj_id = int(match.group(2))

    # Connect via BlitzGateway to resolve names — try session key first, fall back to password
    console.print(f"[dim]Connecting to {server}...[/dim]")
    conn = connect(server, port, session_key=session_key)
    if conn is None:
        conn = connect(server, port, username=username, password=password)
    if conn is None:
        console.print("[red]Could not connect to OMERO. Check credentials/server.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Connected as {conn.getUser().getName()}[/green]")

    try:
        jobs = []  # list of (plate_id, plate_name, output_dir)

        if obj_type == "Screen":
            screen_name, plates = get_plates_for_screen(conn, obj_id)
            screen_folder = output / slugify(screen_name)
            screen_folder.mkdir(parents=True, exist_ok=True)
            console.print(
                f"Screen [bold]{screen_name}[/bold] → {len(plates)} plate(s)"
                f" → [cyan]{screen_folder}[/cyan]"
            )
            for plate_id, plate_name in plates:
                jobs.append((plate_id, plate_name, screen_folder))

        else:  # Plate
            plate_name = get_plate_name(conn, obj_id)
            output.mkdir(parents=True, exist_ok=True)
            console.print(f"Plate [bold]{plate_name}[/bold] → [cyan]{output}[/cyan]")
            jobs.append((obj_id, plate_name, output))

    finally:
        conn.close()

    if not jobs:
        console.print("[yellow]No plates found to export.[/yellow]")
        raise typer.Exit(0)

    # Export each plate sequentially
    success_count = 0
    skip_count = 0

    for i, (plate_id, plate_name, out_dir) in enumerate(jobs, 1):
        console.print(
            f"\n[bold]({i}/{len(jobs)})[/bold] Exporting Plate:{plate_id}"
            f" [dim]({plate_name})[/dim]"
        )

        # --resume: skip plates whose renamed zarr already exists (completed previously)
        safe_name = slugify(plate_name)
        completed_path = out_dir / f"{safe_name}.ome.zarr"
        if resume and completed_path.exists():
            console.print(f"  [dim]Already exported as {completed_path.name}, skipping.[/dim]")
            success_count += 1
            skip_count += 1
            continue

        # Warn if a partial export exists — omero zarr will attempt to resume it
        partial_path = out_dir / f"{plate_id}.ome.zarr"
        if partial_path.exists():
            console.print(f"  [yellow]Partial export found, resuming...[/yellow]")

        ok = run_zarr_export(
            plate_id, out_dir, server, port, username, password,
            tile_width, tile_height
        )
        if not ok:
            continue

        final_path = rename_zarr(out_dir, plate_id, plate_name)
        console.print(f"  → [green]{final_path}[/green]")

        if zip_output:
            zip_path = zip_zarr(final_path)
            console.print(f"  → zipped: [green]{zip_path}[/green]")

        success_count += 1

    console.rule()
    exported = success_count - skip_count
    console.print(
        f"[bold]Done:[/bold] {exported} exported, {skip_count} skipped, "
        f"{len(jobs) - success_count} failed — {success_count}/{len(jobs)} total OK."
    )


if __name__ == "__main__":
    app()
