#!/usr/bin/env python3
"""
GUI Application for ND2 to TIFF Converter
Provides a modern interface for converting multi-position ND2 files to TIFF files
Built with CustomTkinter for a sleek, modern appearance
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import sys
import platform
from pathlib import Path

# Import the conversion function from the converter module
from .converter import convert_nd2_to_tiff_by_well_stack


class TextRedirector:
    """Redirects stdout/stderr to a text widget"""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, text):
        self.widget.configure(state='normal')
        self.widget.insert("end", text, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state='disabled')
        self.widget.update_idletasks()

    def flush(self):
        pass


class ND2ConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ND2 to TIFF Converter")

        # Set appearance mode and color theme FIRST
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        # Platform-specific DPI handling
        # Linux: DPI awareness not implemented in CustomTkinter - need manual scaling
        # Windows/Mac: Automatic DPI detection available but can be problematic
        current_platform = platform.system()

        if current_platform == "Linux":
            # Deactivate automatic DPI awareness and use manual scaling for Linux
            # CRITICAL: Widget scaling only affects widget sizes, NOT font sizes
            # Font sizes must be set independently and much larger
            ctk.deactivate_automatic_dpi_awareness()
            ctk.set_widget_scaling(1.0)  # Modest widget scaling to keep window reasonable
            ctk.set_window_scaling(1.0)
            self.base_font_size = 10  # Very large base font since it doesn't auto-scale
            self.default_scaling = "100%"
        else:
            # Windows and macOS have automatic DPI detection
            ctk.set_widget_scaling(1.0)
            ctk.set_window_scaling(1.0)
            self.base_font_size = 14  # Standard base font for Windows/Mac
            self.default_scaling = "200%"

        # Set window size AFTER scaling
        self.root.geometry("1400x1000")
        self.root.minsize(1000, 800)

        # Variables for file paths
        self.nd2_file = ctk.StringVar()
        self.export_folder = ctk.StringVar()
        self.file_prefix = ctk.StringVar()

        # Variables for options
        self.skip_ome = ctk.BooleanVar(value=False)
        self.separate_channels = ctk.BooleanVar(value=False)
        self.separate_z = ctk.BooleanVar(value=False)
        self.separate_t = ctk.BooleanVar(value=False)
        self.guess_names = ctk.BooleanVar(value=False)
        self.max_projection = ctk.BooleanVar(value=False)

        # Processing state
        self.is_processing = False

        self.create_widgets()

    def create_widgets(self):
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Main container with scrollable frame
        main_frame = ctk.CTkScrollableFrame(self.root, fg_color="transparent")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        main_frame.grid_columnconfigure(0, weight=1)

        current_row = 0

        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="ND2 to TIFF Converter",
            font=ctk.CTkFont(family="Roboto", size=self.base_font_size + 18, weight="bold")
        )
        title_label.grid(row=current_row, column=0, pady=(0, 20))
        current_row += 1

        # Settings Menu Frame (Appearance & Scaling)
        settings_frame = ctk.CTkFrame(main_frame)
        settings_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 15))
        settings_frame.grid_columnconfigure(1, weight=1)
        settings_frame.grid_columnconfigure(3, weight=1)
        current_row += 1

        # Appearance Mode
        ctk.CTkLabel(settings_frame, text="Appearance:",
                    font=ctk.CTkFont(family="Roboto", size=self.base_font_size)).grid(
            row=0, column=0, padx=(15, 5), pady=10, sticky="w"
        )
        self.appearance_menu = ctk.CTkOptionMenu(
            settings_frame,
            values=["System", "Dark", "Light"],
            command=self.change_appearance_mode,
            width=120
        )
        self.appearance_menu.set("System")
        self.appearance_menu.grid(row=0, column=1, padx=(0, 20), pady=10, sticky="w")

        # UI Scaling
        ctk.CTkLabel(settings_frame, text="UI Scale:",
                    font=ctk.CTkFont(family="Roboto", size=self.base_font_size)).grid(
            row=0, column=2, padx=(15, 5), pady=10, sticky="w"
        )
        self.scaling_menu = ctk.CTkOptionMenu(
            settings_frame,
            values=["80%", "90%", "100%", "110%", "120%", "130%", "140%", "150%", "160%", "175%", "200%", "250%", "300%"],
            command=self.change_scaling,
            width=130
        )
        self.scaling_menu.set(self.default_scaling)  # Platform-specific default scaling
        self.scaling_menu.grid(row=0, column=3, padx=(0, 15), pady=10, sticky="w")

        # File selection section
        file_frame = ctk.CTkFrame(main_frame)
        file_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 15))
        file_frame.grid_columnconfigure(1, weight=1)
        current_row += 1

        ctk.CTkLabel(file_frame, text="Input File",
                    font=ctk.CTkFont(family="Roboto", size=self.base_font_size + 6, weight="bold")).grid(
            row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w"
        )

        ctk.CTkLabel(file_frame, text="ND2 File:",
                    font=ctk.CTkFont(family="Roboto", size=self.base_font_size)).grid(
            row=1, column=0, padx=(15, 10), pady=(0, 15), sticky="w"
        )
        self.nd2_entry = ctk.CTkEntry(file_frame, textvariable=self.nd2_file,
                                      placeholder_text="Select an ND2 file...")
        self.nd2_entry.grid(row=1, column=1, padx=(0, 10), pady=(0, 15), sticky="ew")
        ctk.CTkButton(file_frame, text="Browse", command=self.browse_nd2,
                     width=100).grid(row=1, column=2, padx=(0, 15), pady=(0, 15))

        # Output settings section
        output_frame = ctk.CTkFrame(main_frame)
        output_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 15))
        output_frame.grid_columnconfigure(1, weight=1)
        current_row += 1

        ctk.CTkLabel(output_frame, text="Output Settings",
                    font=ctk.CTkFont(family="Roboto", size=self.base_font_size + 6, weight="bold")).grid(
            row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w"
        )

        ctk.CTkLabel(output_frame, text="Export Folder:",
                    font=ctk.CTkFont(family="Roboto", size=self.base_font_size)).grid(
            row=1, column=0, padx=(15, 10), pady=(0, 5), sticky="w"
        )
        self.export_entry = ctk.CTkEntry(output_frame, textvariable=self.export_folder,
                                        placeholder_text="Optional - default is 'export' subfolder")
        self.export_entry.grid(row=1, column=1, padx=(0, 10), pady=(0, 5), sticky="ew")
        ctk.CTkButton(output_frame, text="Browse", command=self.browse_export,
                     width=100).grid(row=1, column=2, padx=(0, 15), pady=(0, 5))

        ctk.CTkLabel(output_frame, text="File Prefix:",
                    font=ctk.CTkFont(family="Roboto", size=self.base_font_size)).grid(
            row=2, column=0, padx=(15, 10), pady=(5, 15), sticky="w"
        )
        self.prefix_entry = ctk.CTkEntry(output_frame, textvariable=self.file_prefix,
                                        placeholder_text="Optional - e.g., '250314_experiment_'")
        self.prefix_entry.grid(row=2, column=1, columnspan=2, padx=(0, 15),
                              pady=(5, 15), sticky="ew")

        # Options section
        options_frame = ctk.CTkFrame(main_frame)
        options_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 15))
        current_row += 1

        ctk.CTkLabel(options_frame, text="Conversion Options",
                    font=ctk.CTkFont(family="Roboto", size=self.base_font_size + 6, weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=15, pady=(15, 10), sticky="w"
        )

        # Create two columns for options
        options_left = ctk.CTkFrame(options_frame, fg_color="transparent")
        options_left.grid(row=1, column=0, padx=(15, 10), pady=(0, 15), sticky="nsew")

        options_right = ctk.CTkFrame(options_frame, fg_color="transparent")
        options_right.grid(row=1, column=1, padx=(10, 15), pady=(0, 15), sticky="nsew")

        options_frame.grid_columnconfigure(0, weight=1)
        options_frame.grid_columnconfigure(1, weight=1)

        # Left column options with larger font
        ctk.CTkCheckBox(options_left, text="Skip OME metadata",
                       variable=self.skip_ome,
                       font=ctk.CTkFont(family="Roboto", size=self.base_font_size)).pack(anchor="w", pady=5)
        ctk.CTkCheckBox(options_left, text="Separate channels",
                       variable=self.separate_channels,
                       font=ctk.CTkFont(family="Roboto", size=self.base_font_size)).pack(anchor="w", pady=5)
        ctk.CTkCheckBox(options_left, text="Separate Z-slices",
                       variable=self.separate_z,
                       font=ctk.CTkFont(family="Roboto", size=self.base_font_size)).pack(anchor="w", pady=5)

        # Right column options with larger font
        ctk.CTkCheckBox(options_right, text="Separate time points",
                       variable=self.separate_t,
                       font=ctk.CTkFont(family="Roboto", size=self.base_font_size)).pack(anchor="w", pady=5)
        ctk.CTkCheckBox(options_right, text="Guess missing position names",
                       variable=self.guess_names,
                       font=ctk.CTkFont(family="Roboto", size=self.base_font_size)).pack(anchor="w", pady=5)
        ctk.CTkCheckBox(options_right, text="Max projection only",
                       variable=self.max_projection,
                       font=ctk.CTkFont(family="Roboto", size=self.base_font_size)).pack(anchor="w", pady=5)

        # Control buttons
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.grid(row=current_row, column=0, pady=(0, 15))
        current_row += 1

        self.convert_button = ctk.CTkButton(
            button_frame,
            text="Convert",
            command=self.start_conversion,
            width=140,
            height=36,
            font=ctk.CTkFont(family="Roboto", size=self.base_font_size, weight="bold")
        )
        self.convert_button.grid(row=0, column=0, padx=10)

        ctk.CTkButton(
            button_frame,
            text="Clear Log",
            command=self.clear_log,
            width=140,
            height=36
        ).grid(row=0, column=1, padx=10)

        ctk.CTkButton(
            button_frame,
            text="Exit",
            command=self.root.quit,
            width=140,
            height=36,
            fg_color="#d32f2f",
            hover_color="#b71c1c"
        ).grid(row=0, column=2, padx=10)

        # Progress/Log section
        log_frame = ctk.CTkFrame(main_frame)
        log_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 15))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        current_row += 1

        ctk.CTkLabel(log_frame, text="Progress Log",
                    font=ctk.CTkFont(family="Roboto", size=self.base_font_size + 6, weight="bold")).grid(
            row=0, column=0, padx=15, pady=(15, 10), sticky="w"
        )

        self.log_text = ctk.CTkTextbox(log_frame, height=250, state='disabled',
                                       wrap="word",
                                       font=ctk.CTkFont(family="Courier", size=self.base_font_size - 1))
        self.log_text.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="ew")

        # Configure text tags for colored output
        self.log_text.tag_config("stdout", foreground="#00ff00")
        self.log_text.tag_config("stderr", foreground="#ff4444")

        # Status bar
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Ready",
            font=ctk.CTkFont(family="Roboto", size=self.base_font_size),
            anchor="w"
        )
        self.status_label.grid(row=current_row, column=0, sticky="ew", pady=(0, 0))

    def change_appearance_mode(self, new_mode: str):
        """Change the appearance mode"""
        ctk.set_appearance_mode(new_mode.lower())

    def change_scaling(self, new_scaling: str):
        """Change the UI scaling"""
        scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(scaling_float)
        ctk.set_window_scaling(scaling_float)

    def browse_nd2(self):
        """Open file dialog to select ND2 file"""
        filename = filedialog.askopenfilename(
            title="Select ND2 File",
            filetypes=[("ND2 files", "*.nd2"), ("All files", "*.*")]
        )
        if filename:
            self.nd2_file.set(filename)
            # Auto-set export folder to default if not already set
            if not self.export_folder.get():
                default_export = Path(filename).parent / "export"
                self.export_folder.set(str(default_export))

    def browse_export(self):
        """Open directory dialog to select export folder"""
        dirname = filedialog.askdirectory(title="Select Export Folder")
        if dirname:
            self.export_folder.set(dirname)

    def clear_log(self):
        """Clear the log text widget"""
        self.log_text.configure(state='normal')
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state='disabled')

    def log_message(self, message, tag="stdout"):
        """Add a message to the log"""
        self.log_text.configure(state='normal')
        self.log_text.insert("end", message + "\n", (tag,))
        self.log_text.see("end")
        self.log_text.configure(state='disabled')

    def validate_inputs(self):
        """Validate user inputs before conversion"""
        if not self.nd2_file.get():
            messagebox.showerror("Error", "Please select an ND2 file")
            return False

        nd2_path = Path(self.nd2_file.get())
        if not nd2_path.exists():
            messagebox.showerror("Error", f"File does not exist: {nd2_path}")
            return False

        if not nd2_path.suffix.lower() == '.nd2':
            messagebox.showerror("Error", "Selected file is not an ND2 file")
            return False

        return True

    def start_conversion(self):
        """Start the conversion process in a separate thread"""
        if self.is_processing:
            messagebox.showwarning("Warning", "Conversion is already in progress")
            return

        if not self.validate_inputs():
            return

        # Disable the convert button
        self.convert_button.configure(state='disabled', text="Converting...")
        self.is_processing = True
        self.status_label.configure(text="Converting...")

        # Start conversion in a separate thread
        thread = threading.Thread(target=self.run_conversion, daemon=True)
        thread.start()

    def run_conversion(self):
        """Run the conversion process"""
        # Redirect stdout and stderr to the log widget
        sys.stdout = TextRedirector(self.log_text, "stdout")
        sys.stderr = TextRedirector(self.log_text, "stderr")

        try:
            # Get the export folder, use None if empty (will use default)
            export_folder = self.export_folder.get() if self.export_folder.get() else None
            file_prefix = self.file_prefix.get() if self.file_prefix.get() else None

            self.log_message("="*60)
            self.log_message("Starting conversion...")
            self.log_message("="*60)

            # Call the conversion function
            convert_nd2_to_tiff_by_well_stack(
                self.nd2_file.get(),
                skip_ome=self.skip_ome.get(),
                separate_channels=self.separate_channels.get(),
                separate_z=self.separate_z.get(),
                separate_t=self.separate_t.get(),
                guess_names=self.guess_names.get(),
                max_projection=self.max_projection.get(),
                export_folder=export_folder,
                file_prefix=file_prefix
            )

            self.log_message("="*60)
            self.log_message("Conversion completed successfully!")
            self.log_message("="*60)

            # Show success message
            self.root.after(0, lambda: messagebox.showinfo("Success", "Conversion completed successfully!"))
            self.root.after(0, lambda: self.status_label.configure(text="Conversion completed"))

        except Exception as e:
            error_msg = f"Error during conversion: {str(e)}"
            self.log_message(error_msg, "stderr")
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.status_label.configure(text="Conversion failed"))

        finally:
            # Restore stdout and stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

            # Re-enable the convert button
            self.root.after(0, lambda: self.convert_button.configure(state='normal', text="Convert"))
            self.is_processing = False


def main():
    root = ctk.CTk()
    app = ND2ConverterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
