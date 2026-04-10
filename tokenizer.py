import os
import sys
import time

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text
    import tiktoken
    import fitz  # PyMuPDF
except ImportError:
    print("Missing required dependencies. Please install them by running:")
    print("pip install rich tiktoken pymupdf")
    sys.exit(1)

console = Console()

def extract_text_from_file(file_path: str) -> str:
    """Extracts text from .txt, .md, or .pdf files. Converts PDF to Markdown."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
            
    elif ext == '.pdf':
        text_content = []
        try:
            doc = fitz.open(file_path)
            for page in doc:
                # PyMuPDF >= 1.24.0 natively supports markdown text extraction
                try:
                    text_content.append(page.get_text("markdown"))
                except ValueError:
                    # Fallback for older versions of PyMuPDF
                    text_content.append(page.get_text("text"))
            return "\n\n".join(text_content)
        except Exception as e:
            console.print(f"[red]Error reading PDF: {e}[/red]")
            sys.exit(1)
    else:
        console.print(f"[red]Unsupported file format: {ext}. Please use .txt, .md, or .pdf[/red]")
        sys.exit(1)

def get_token_estimates(text: str) -> dict:
    """
    Estimates token counts. Uses exact tiktoken encodings where applicable, 
    and fast heuristic proxies for other models to ensure CPU speed.
    """
    results = []
    
    # 1. GPT-5.x / GPT-4o (Exact: o200k_base)
    try:
        enc_o200k = tiktoken.get_encoding("o200k_base")
    except ValueError:
        console.print("[yellow]Warning: tiktoken update recommended for o200k_base. Falling back to cl100k_base.[/yellow]")
        enc_o200k = tiktoken.get_encoding("cl100k_base")
        
    gpt5_tokens = len(enc_o200k.encode(text, disallowed_special=()))
    results.append({
        "family": "GPT-5.x / GPT-4o",
        "method": "Exact (o200k_base)",
        "tokens": gpt5_tokens,
        "baseline": gpt5_tokens
    })

    # 2. Claude Sonnet 4.x (Proxy: cl100k_base)
    # Anthropic recommends cl100k_base as a standard offline proxy.
    enc_cl100k = tiktoken.get_encoding("cl100k_base")
    cl_base_tokens = len(enc_cl100k.encode(text, disallowed_special=()))
    claude_tokens = int(cl_base_tokens * 1.01) # +1% buffer for Anthropic's parsing overhead
    results.append({
        "family": "Claude Sonnet 4.x / 3.5",
        "method": "Proxy (cl100k_base + 1%)",
        "tokens": claude_tokens,
        "baseline": gpt5_tokens
    })

    # 3. Gemini 3.x / 1.5 (Proxy: SentencePiece Heuristic)
    # Google's tokenizers typically yield ~10-15% more tokens than OpenAI's dense o200k_base.
    gemini_tokens = int(gpt5_tokens * 1.14)
    results.append({
        "family": "Gemini 3.x / 1.5",
        "method": "Estimate (o200k * 1.14)",
        "tokens": gemini_tokens,
        "baseline": gpt5_tokens
    })

    # 4. Open Source: Llama 3.x
    # Llama 3 has a 128k vocab. Very similar efficiency to OpenAI, slightly higher token count.
    llama_tokens = int(gpt5_tokens * 1.05)
    results.append({
        "family": "Open Source (Llama-3.x)",
        "method": "Estimate (128k Vocab Proxy)",
        "tokens": llama_tokens,
        "baseline": gpt5_tokens
    })

    # 5. Open Source: Mistral / Legacy
    # Mistral uses a 32k vocab, which splits words much more often.
    mistral_tokens = int(cl_base_tokens * 1.18)
    results.append({
        "family": "Open Source (Mistral-7B/8x7B)",
        "method": "Estimate (32k Vocab Proxy)",
        "tokens": mistral_tokens,
        "baseline": gpt5_tokens
    })

    return results

def display_results(filename: str, file_size_kb: float, word_count: int, estimates: list):
    """Renders a beautiful Rich table with the results."""
    table = Table(
        title=f"\nToken Estimation Analysis\n[cyan]{filename}[/cyan] ({file_size_kb:.1f} KB | ~{word_count:,} words)", 
        border_style="cyan",
        title_justify="center",
        box=import_box()
    )
    
    table.add_column("Model Family", justify="left", style="white bold", no_wrap=True)
    table.add_column("Tokenizer / Method", style="dim")
    table.add_column("Est. Tokens", justify="right", style="magenta bold")
    table.add_column("Vs. GPT-5", justify="right")

    for est in estimates:
        # Calculate ratio compared to GPT-5
        ratio = (est['tokens'] / est['baseline']) * 100 if est['baseline'] > 0 else 100
        
        # Color code the ratio
        if ratio == 100:
            ratio_str = "[dim]Baseline[/dim]"
        elif ratio > 100:
            ratio_str = f"[yellow]+{ratio - 100:.1f}%[/yellow]"
        else:
            ratio_str = f"[green]{ratio - 100:.1f}%[/green]"

        table.add_row(
            est['family'],
            est['method'],
            f"{est['tokens']:,}",
            ratio_str
        )

    console.print(table)
    console.print()

def import_box():
    from rich import box
    return box.ROUNDED

def main():
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]🚀 Universal Fast Token Estimator[/bold cyan]\n"
        "[dim]Estimates tokens for GPT-5, Claude 4, Gemini 3, and Open Source models.[/dim]\n"
        "[dim]Supports .txt, .md, and automatic .pdf to markdown conversion.[/dim]",
        border_style="cyan"
    ))

    while True:
        try:
            file_path = Prompt.ask("\n[bold green]?[/bold green] Enter the path to your file (or 'q' to quit)").strip()
            
            if file_path.lower() in ['q', 'quit', 'exit']:
                break
                
            # Strip quotes if dragged-and-dropped into terminal
            file_path = file_path.strip("\"'")

            if not os.path.exists(file_path):
                console.print(f"[red]✖ File not found: {file_path}[/red]")
                continue

            with console.status("[bold cyan]Processing document and counting tokens...", spinner="dots"):
                # 1. Extract Text
                start_time = time.time()
                text = extract_text_from_file(file_path)
                
                # Metrics
                file_size_kb = os.path.getsize(file_path) / 1024
                word_count = len(text.split())
                
                # 2. Estimate Tokens
                estimates = get_token_estimates(text)
                elapsed = time.time() - start_time

            # 3. Display
            display_results(os.path.basename(file_path), file_size_kb, word_count, estimates)
            console.print(f"[dim]⚡ Processed instantly on CPU in {elapsed:.3f} seconds.[/dim]\n")

        except KeyboardInterrupt:
            console.print("\n[dim]Exiting...[/dim]")
            break
        except Exception as e:
            console.print(f"\n[red]An unexpected error occurred: {e}[/red]")

if __name__ == "__main__":
    main()