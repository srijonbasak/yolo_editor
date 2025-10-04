from __future__ import annotations
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from .services import configure_services, get_service, ILogger, IConfigService
from .ui.main_window import MainWindow


def setup_application() -> QApplication:
    """Set up the Qt application with proper configuration."""
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Editor")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("YOLO Editor")
    app.setOrganizationDomain("yolo-editor.com")
    
    return app


def setup_services() -> None:
    """Set up the service container with all dependencies."""
    # Determine log file location
    log_dir = Path.home() / ".yolo-editor" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "yolo-editor.log"
    
    # Configure services
    configure_services(log_file=log_file)
    
    # Get logger and log startup
    logger = get_service(ILogger)
    logger.info("YOLO Editor starting up")
    logger.info(f"Log file: {log_file}")
    
    # Log system information
    import platform
    logger.log_system_info({
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "qt_version": "PySide6"  # Could get actual version
    })


def main():
    """Main application entry point."""
    try:
        # Set up Qt application
        app = setup_application()
        
        # Set up services
        setup_services()
        
        # Get services
        logger = get_service(ILogger)
        config_service = get_service(IConfigService)
        
        # Load configuration
        config = config_service.load_config()
        logger.debug("Configuration loaded")
        
        # Create and show main window
        main_window = MainWindow()
        
        # Apply window configuration
        ui_config = config_service.get_ui_config()
        main_window.resize(ui_config.window_width, ui_config.window_height)
        
        main_window.show()
        logger.info("Main window displayed")
        
        # Set up graceful shutdown
        def on_about_to_quit():
            logger.info("Application shutting down")
            # Save any pending configuration changes
            config_service.save_config(config_service.load_config())
        
        app.aboutToQuit.connect(on_about_to_quit)
        
        # Start the event loop
        logger.info("Starting Qt event loop")
        exit_code = app.exec()
        
        logger.info(f"Application exited with code: {exit_code}")
        return exit_code
        
    except Exception as e:
        # Try to log the error if possible
        try:
            logger = get_service(ILogger)
            logger.error("Fatal error during application startup", exception=e)
        except:
            print(f"Fatal error during application startup: {e}", file=sys.stderr)
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
