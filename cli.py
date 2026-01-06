"""
Factor CLI - Command line interface for testing and utilities.

Usage:
    python cli.py extract <file_path> [--source-type TYPE]
    python cli.py stats
    python cli.py clear
"""

import asyncio
import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from src.config import Config
from src.storage.repository import FactRepository
from src.storage.models import SourceType, RelevanceLevel, Believability, STANDARD_CHAPTERS
from src.extraction.pdf_parser import extract_pdf
from src.extraction.excel_parser import extract_excel
from src.extraction.fact_extractor import FactExtractor
from src.utils.embeddings import deduplicate_facts
from src.scoring.scorer import get_chapter_summary

console = Console()


def cmd_stats(args):
    """Display database statistics."""
    repo = FactRepository()
    stats = repo.get_statistics()
    
    # Header
    console.print(Panel.fit(
        "[bold blue]Factor Fact Bank Statistics[/bold blue]",
        border_style="blue"
    ))
    
    # Main stats table
    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Total Documents", str(stats['total_documents']))
    stats_table.add_row("Total Facts", str(stats['total_facts']))
    
    console.print(stats_table)
    console.print()
    
    # Relevance breakdown
    if stats['total_facts'] > 0:
        relevance_table = Table(title="Relevance Breakdown")
        relevance_table.add_column("Level", style="cyan")
        relevance_table.add_column("Count", style="green")
        relevance_table.add_column("Percentage", style="yellow")
        
        relevance_breakdown = stats.get('relevance_breakdown', {})
        total = sum(relevance_breakdown.values())
        
        relevance_display = [
            ('critical', 'Critical', 'red'),
            ('chapter_specific', 'Chapter Specific', 'blue'),
            ('additional_context', 'Additional Context', 'dim'),
            ('noise', 'Noise', 'dim'),
        ]
        
        for key, display_name, color in relevance_display:
            count = relevance_breakdown.get(key, 0)
            pct = count / total * 100 if total > 0 else 0
            relevance_table.add_row(
                f"[{color}]{display_name}[/{color}]",
                str(count),
                f"{pct:.1f}%"
            )
        
        console.print(relevance_table)
        console.print()
        
        # Believability breakdown
        believability_table = Table(title="Believability Breakdown")
        believability_table.add_column("Level", style="cyan")
        believability_table.add_column("Count", style="green")
        believability_table.add_column("Percentage", style="yellow")
        
        believability_breakdown = stats.get('believability_breakdown', {})
        total_bel = sum(believability_breakdown.values())
        
        believability_display = [
            ('verified', 'Verified', 'green'),
            ('needs_verification', 'Needs Verification', 'yellow'),
            ('opinion', 'Opinion', 'magenta'),
        ]
        
        for key, display_name, color in believability_display:
            count = believability_breakdown.get(key, 0)
            pct = count / total_bel * 100 if total_bel > 0 else 0
            believability_table.add_row(
                f"[{color}]{display_name}[/{color}]",
                str(count),
                f"{pct:.1f}%"
            )
        
        console.print(believability_table)
        console.print()
        
        # Chapter summary
        facts = repo.get_all_facts()
        chapter_stats = get_chapter_summary(facts)
        
        chapter_table = Table(title="Chapter Coverage")
        chapter_table.add_column("Chapter", style="cyan")
        chapter_table.add_column("Facts", style="green")
        chapter_table.add_column("Critical", style="red")
        chapter_table.add_column("Ch.Specific", style="blue")
        chapter_table.add_column("Context", style="dim")
        chapter_table.add_column("Verified", style="green")
        
        for chapter in STANDARD_CHAPTERS:
            cs = chapter_stats[chapter]
            rel = cs.get('relevance_breakdown', {})
            bel = cs.get('believability_breakdown', {})
            chapter_table.add_row(
                chapter,
                str(cs['fact_count']),
                str(rel.get('critical', 0)),
                str(rel.get('chapter_specific', 0)),
                str(rel.get('additional_context', 0)),
                str(bel.get('verified', 0)),
            )
        
        console.print(chapter_table)


async def async_extract(file_path: Path, source_type: SourceType, run_dedup: bool):
    """Async extraction logic."""
    repo = FactRepository()
    
    # Extract document
    console.print(f"[cyan]Extracting content from:[/cyan] {file_path.name}")
    
    file_ext = file_path.suffix.lower()
    if file_ext == ".pdf":
        doc = extract_pdf(file_path)
    elif file_ext in (".xlsx", ".xls"):
        doc = extract_excel(file_path)
    else:
        console.print(f"[red]Unsupported file type:[/red] {file_ext}")
        return
    
    console.print(f"[green]✓[/green] Extracted {len(doc.sections)} sections")
    
    # Extract facts
    console.print("[cyan]Extracting facts with LLM...[/cyan]")
    
    extractor = FactExtractor()
    try:
        facts = await extractor.extract_facts(doc, source_type)
        console.print(f"[green]✓[/green] Extracted {len(facts)} facts")
        
        # Deduplication
        if run_dedup and len(facts) > 1:
            console.print("[cyan]Running deduplication...[/cyan]")
            original_count = len(facts)
            facts = await deduplicate_facts(facts)
            merged = original_count - len(facts)
            if merged > 0:
                console.print(f"[green]✓[/green] Merged {merged} duplicate facts")
        
        # Store in database
        repo.insert_facts(facts)
        console.print(f"[green]✓[/green] Stored {len(facts)} facts in database")
        
            # Display sample
            if facts:
                console.print()
                sample_table = Table(title="Sample Extracted Facts (first 5)")
                sample_table.add_column("Content", style="white", max_width=60)
                sample_table.add_column("Relevance", style="cyan")
                sample_table.add_column("Believability", style="green")
                
                relevance_display = {
                    'critical': 'Critical',
                    'chapter_specific': 'Ch.Specific',
                    'additional_context': 'Context',
                    'noise': 'Noise',
                }
                
                believability_display = {
                    'verified': 'Verified',
                    'needs_verification': 'NeedsVerif',
                    'opinion': 'Opinion',
                }
                
                for fact in facts[:5]:
                    content = fact.content[:80] + "..." if len(fact.content) > 80 else fact.content
                    sample_table.add_row(
                        content,
                        relevance_display.get(fact.relevance, fact.relevance),
                        believability_display.get(fact.believability, fact.believability),
                    )
                
                console.print(sample_table)
            
    finally:
        await extractor.close()


def cmd_extract(args):
    """Extract facts from a document."""
    file_path = Path(args.file_path)
    
    if not file_path.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        sys.exit(1)
    
    # Validate API key
    config_issues = Config.validate()
    if config_issues:
        console.print("[red]Configuration Error:[/red]")
        for issue in config_issues:
            console.print(f"  - {issue}")
        console.print("\nPlease set up your .env file with required API keys.")
        sys.exit(1)
    
    # Parse source type
    source_type_map = {
        "presentation": SourceType.COMPANY_PRESENTATION,
        "company": SourceType.COMPANY_PRESENTATION,
        "market": SourceType.MARKET_RESEARCH,
        "research": SourceType.MARKET_RESEARCH,
        "financial": SourceType.FINANCIALS,
        "financials": SourceType.FINANCIALS,
    }
    
    source_type = source_type_map.get(
        args.source_type.lower(),
        SourceType.COMPANY_PRESENTATION
    )
    
    # Run extraction
    asyncio.run(
        async_extract(file_path, source_type, not args.no_dedup)
    )


def cmd_clear(args):
    """Clear all data from the database."""
    repo = FactRepository()
    
    if not args.yes:
        confirm = input("Are you sure you want to clear all facts? [y/N] ")
        if confirm.lower() != 'y':
            console.print("[yellow]Cancelled[/yellow]")
            return
    
    repo.clear_all_facts()
    console.print("[green]✓[/green] Database cleared")


def cmd_list(args):
    """List facts in the database."""
    repo = FactRepository()
    facts = repo.get_all_facts()
    
    if not facts:
        console.print("[yellow]No facts in database[/yellow]")
        return
    
    # Apply filters
    if args.relevance:
        facts = [f for f in facts if f.relevance == args.relevance.lower()]
    
    if args.believability:
        facts = [f for f in facts if f.believability == args.believability.lower()]
    
    if args.chapter:
        facts = [f for f in facts if f.chapter_relevance.get(args.chapter, 0) >= 0.5]
    
    if args.search:
        search_lower = args.search.lower()
        facts = [f for f in facts if search_lower in f.content.lower()]
    
    # Filter out noise by default unless explicitly requested
    if not args.show_noise and not args.relevance:
        facts = [f for f in facts if f.relevance != 'noise']
    
    # Limit
    limit = args.limit or 20
    facts = facts[:limit]
    
    # Relevance abbreviations
    rel_abbrev = {
        'critical': 'Crit',
        'chapter_specific': 'Ch.',
        'additional_context': 'Ctx',
        'noise': 'Noi',
    }
    
    bel_abbrev = {
        'verified': 'Ver',
        'needs_verification': 'NV',
        'opinion': 'Op',
    }
    
    # Display
    table = Table(title=f"Facts ({len(facts)} shown)")
    table.add_column("ID", style="dim", width=8)
    table.add_column("Content", style="white", max_width=50)
    table.add_column("Source", style="cyan", max_width=15)
    table.add_column("Rel", style="yellow", width=5)
    table.add_column("Bel", style="green", width=4)
    
    for fact in facts:
        content = fact.content[:60] + "..." if len(fact.content) > 60 else fact.content
        source = fact.source_documents[0][:15] if fact.source_documents else ""
        table.add_row(
            fact.id[:8],
            content,
            source,
            rel_abbrev.get(fact.relevance, fact.relevance[:3]),
            bel_abbrev.get(fact.believability, fact.believability[:3]),
        )
    
    console.print(table)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Factor CLI - Fact Bank Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract facts from a document")
    extract_parser.add_argument("file_path", help="Path to PDF or Excel file")
    extract_parser.add_argument(
        "--source-type", "-t",
        default="presentation",
        choices=["presentation", "company", "market", "research", "financial", "financials"],
        help="Type of source document"
    )
    extract_parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip deduplication"
    )
    
    # Stats command
    subparsers.add_parser("stats", help="Display database statistics")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List facts in database")
    list_parser.add_argument(
        "--relevance", "-r",
        choices=["critical", "chapter_specific", "additional_context", "noise"],
        help="Filter by relevance level"
    )
    list_parser.add_argument(
        "--believability", "-b",
        choices=["verified", "needs_verification", "opinion"],
        help="Filter by believability level"
    )
    list_parser.add_argument("--chapter", "-c", choices=STANDARD_CHAPTERS)
    list_parser.add_argument("--search", "-s", help="Search in content")
    list_parser.add_argument("--limit", "-n", type=int, help="Max facts to show")
    list_parser.add_argument(
        "--show-noise", 
        action="store_true",
        help="Include noise facts (hidden by default)"
    )
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all data from database")
    clear_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "clear":
        cmd_clear(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

