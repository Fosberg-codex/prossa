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

async def main():
    # Example usage with error handling
    try:
        # Process single dataset
        logger.info("Processing military dataset example...")
        await process_dataset(
            Path("dataset/military.csv"),
            Agent(),
            ReportGenerator()
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 