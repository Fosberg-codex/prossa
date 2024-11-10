from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from datetime import datetime
import json
from pathlib import Path

class ReportGenerator:
    def __init__(self, output_directory: str = "./reports"):
        """Initialize the report generator"""
        self.console = Console()
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, 
                       analysis_results: Dict[str, Any],
                       format: str = "console") -> None:
        """Generate and output the analysis report in specified format"""
        if format == "console":
            self._generate_console_report(analysis_results)
        elif format == "json":
            self._save_json_report(analysis_results)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_console_report(self, results: Dict[str, Any]) -> None:
        """Generate a rich console report"""
        # Dataset Overview
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold blue]Prossa Dataset Analysis Report[/bold blue]",
            subtitle=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ))
        
        # Dataset Metadata
        self._print_dataset_metadata(results["dataset_metadata"])
        
        # Recommendations Summary
        self._print_recommendations_summary(results["summary"])
        
        # Detailed Recommendations
        self._print_detailed_recommendations(results["recommendations"])
    
    def _print_dataset_metadata(self, metadata: Dict[str, Any]) -> None:
        """Print dataset metadata section"""
        metadata_table = Table(title="Dataset Overview", show_header=True)
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value", style="green")
        
        metadata_table.add_row("Dataset Type", metadata["type"])
        metadata_table.add_row("Dimensions", f"{metadata['shape'][0]} rows × {metadata['shape'][1]} columns")
        metadata_table.add_row("Complexity Score", f"{metadata['complexity']:.2f}")
        
        self.console.print("\n")
        self.console.print(metadata_table)
    
    def _print_recommendations_summary(self, summary: Dict[str, Any]) -> None:
        """Print recommendations summary section"""
        summary_table = Table(title="Recommendations Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Recommendations", str(summary["total_recommendations"]))
        summary_table.add_row("Average Confidence", f"{summary['average_confidence']:.2%}")
        summary_table.add_row("Recommendation Types", ", ".join(summary["types"]))
        
        self.console.print("\n")
        self.console.print(summary_table)
    
    def _print_detailed_recommendations(self, recommendations: List[Dict[str, Any]]) -> None:
        """Print detailed recommendations section"""
        self.console.print("\n")
        self.console.print("[bold blue]Detailed Recommendations[/bold blue]")
        
        for i, rec in enumerate(recommendations, 1):
            self._print_recommendation(rec, i)
    
    def _print_recommendation(self, recommendation: Dict[str, Any], index: int) -> None:
        """Print individual recommendation details"""
        content = recommendation["recommendation"]["content"]
        validation = recommendation["validation"]
        
        rec_panel = Panel(
            Markdown(f"""
### {index}. {recommendation['type'].title()} Recommendation

{content}

**Confidence Score**: {validation['confidence_score']:.2%}
**Model**: {recommendation['recommendation']['model']}
            """.strip()),
            title=f"Recommendation {index}",
            border_style="blue" if validation["confidence_score"] >= 0.9 else "yellow"
        )
        
        self.console.print("\n")
        self.console.print(rec_panel)
    
    def _save_json_report(self, results: Dict[str, Any]) -> None:
        """Save report as JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"prossa_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.console.print(f"\n[green]Report saved to: {filename}[/green]")
    
    def export_report(self, 
                     results: Dict[str, Any],
                     format: str = "json",
                     filename: Optional[str] = None) -> None:
        """Export report in specified format"""
        if format == "json":
            if filename is None:
                filename = f"prossa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.console.print(f"\n[green]Report exported to: {filepath}[/green]")
        else:
            raise ValueError(f"Unsupported export format: {format}") 