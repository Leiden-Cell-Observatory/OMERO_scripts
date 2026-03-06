"""
omero_zarr_export.py

Export plates from OMERO as OME-Zarr files, optionally discovering all plates
within a screen. Wraps the omero-cli-zarr plugin (`omero zarr export`) and
handles naming, folder structure, and optional zipping.

Usage examples:
  python omero_zarr_export.py
  python omero_zarr_export.py --object Screen:5 --output ./zarr_exports
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
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_current_session():
    """
    Read the active omero CLI session from ~/.omero/sessions (same as omero CLI).
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
            password: str = None, session_key: str = None) -> BlitzGateway:
    """Connect via password or reuse an existing session key."""
    if session_key:
        conn = BlitzGateway(host=server, port=port, secure=True)
        conn.connect(sUuid=session_key)
    else:
        conn = BlitzGateway(username, password, host=server, port=port, secure=True)
        conn.connect()
    if not conn.isConnected():
        console.print("[red]Could not connect to OMERO. Check credentials/server.[/red]")
        raise typer.Exit(1)
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
    session_key: str = None,
) -> bool:
    """
    Call `omero zarr export Plate:ID` as a subprocess.
    Returns True on success, False on failure.
    The output will be written as <output_dir>/<plate_id>.ome.zarr.
    """
    if session_key:
        auth_args = ["-s", server, "-k", session_key]
    else:
        auth_args = ["-s", server, "-u", username, "-w", password]

    # Only pass port explicitly if non-default to avoid "port specified twice" errors
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

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Export failed for Plate:{plate_id}[/red]")
        console.print(result.stderr)
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
    zip_output: bool = typer.Option(False, "--zip", help="Zip each zarr folder after export"),
    tile_width: Optional[int] = typer.Option(None, "--tile-width", help="Zarr chunk tile width"),
    tile_height: Optional[int] = typer.Option(None, "--tile-height", help="Zarr chunk tile height"),
):
    """Export OMERO plates (or all plates in a screen) as OME-Zarr files."""

    console.rule("[bold blue]OMERO Zarr Export[/bold blue]")

    # Try to reuse an existing omero CLI session (same store as `omero login`)
    sess_server, sess_user, session_key, sess_port = get_current_session()

    if session_key and not any([server, username, password]):
        # Reuse existing session — no credentials needed
        console.print(f"[dim]Reusing session for {sess_user}@{sess_server}[/dim]")
        server = sess_server
        username = sess_user
        port = sess_port
    else:
        # Ignore stored session if credentials were explicitly provided
        session_key = None
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

    # Connect via BlitzGateway to resolve screen/plate names before exporting
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        progress.add_task("Connecting to OMERO...")
        conn = connect(server, port, username, password, session_key)

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
    for i, (plate_id, plate_name, out_dir) in enumerate(jobs, 1):
        console.print(
            f"\n[bold]({i}/{len(jobs)})[/bold] Exporting Plate:{plate_id}"
            f" [dim]({plate_name})[/dim]"
        )

        ok = run_zarr_export(
            plate_id, out_dir, server, port, username, password,
            tile_width, tile_height, session_key
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
    console.print(
        f"[bold]Done:[/bold] {success_count}/{len(jobs)} plates exported successfully."
    )


if __name__ == "__main__":
    app()
