import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich import box

# Create a console for Rich
console = Console()

# Configure logging with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)]
)

# Create a logger
logger = logging.getLogger("plato")


# Function to log informative messages
def log_info(message):
    logger.info(f"[bold green]INFO:[/bold green] {message}")


# Function to log warnings
def log_warning(message):
    logger.warning(f"[bold yellow]WARNING:[/bold yellow] {message}")


# Function to log errors
def log_error(message):
    logger.error(f"[bold red]ERROR:[/bold red] {message}")


# Function to log debug messages
def log_debug(message):
    logger.debug(f"[bold blue]DEBUG:[/bold blue] {message}")


# Function to log analysis lifecycle steps
def log_lifecycle_step(step, description):
    panel = Panel.fit(
        f"[bold magenta]{step}[/bold magenta]: {description}",
        box=box.ROUNDED,
        style="magenta"
    )
    console.print(panel)


# Example usage in the data analysis lifecycle
if __name__ == "__main__":
    log_info("Starting the data analysis process.")
    log_lifecycle_step("Data Ingestion", "Loading data from CSV files")
    log_info("Data loaded successfully.")

    log_lifecycle_step("Data Transformation", "Cleaning and transforming data")
    log_warning("Missing values found in the dataset.")

    log_lifecycle_step("Data Analysis", "Performing quantitative analysis")
    log_error("Error during hypothesis testing.")

    log_lifecycle_step("Reporting", "Generating analysis report")
    log_debug("Report generated successfully but with some warnings.")
