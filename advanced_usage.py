import pandas as pd
import asyncio
from pathlib import Path
import logging
from prossa_agent.core.agent import Agent
from prossa_agent.utils.reporting import ReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_dataset(filepath: Path, agent: Agent, report_generator: ReportGenerator) -> None:
    """Process a single dataset with detailed error handling"""
    try:
        # Load dataset
        df = pd.read_csv(filepath)
        dataset_name = filepath.stem
        
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"Shape: {df.shape}")
        
        # Analyze dataset
        analysis_results = await agent.analyze_dataset(
            dataset=df,
            dataset_name=dataset_name
        )
        
        # Generate reports
        report_generator.generate_report(
            analysis_results=analysis_results,
            format="console"
        )
        
        # Export JSON report
        report_generator.export_report(
            results=analysis_results,
            format="json",
            filename=f"{dataset_name}_report.json"
        )
        
        logger.info(f"Successfully processed {dataset_name}")
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        raise

async def batch_process_datasets(data_directory: str) -> None:
    """Process multiple datasets in a directory"""
    data_dir = Path(data_directory)
    
    # Initialize components
    agent = Agent(persist_directory="./prossa_data")
    report_generator = ReportGenerator(output_directory="./analysis_reports")
    
    # Process all CSV files in directory
    csv_files = list(data_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    for filepath in csv_files:
        await process_dataset(filepath, agent, report_generator)

async def main():
    # Example usage with error handling
    try:
        # Process single dataset
        logger.info("Processing single dataset example...")
        await process_dataset(
            Path("data/sample.csv"),
            Agent(),
            ReportGenerator()
        )
        
        # Process multiple datasets
        logger.info("\nProcessing batch datasets example...")
        await batch_process_datasets("data/batch_datasets")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 