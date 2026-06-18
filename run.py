import argparse
from datetime import datetime
import agent_core
from agent_core import run_sixt, run_enterprise, save_results


def main():
    parser = argparse.ArgumentParser(description="Universal Car Rental Scraper")
    parser.add_argument("--site", required=True, choices=["sixt", "enterprise"], 
                       help="Rental site to scrape")
    parser.add_argument("--location", required=True, 
                       help="Pickup location (e.g., 'Toronto Airport')")
    parser.add_argument("--pickup-date", 
                       help="Pickup date (YYYY-MM-DD). Defaults to tomorrow")
    parser.add_argument("--return-date", 
                       help="Return date (YYYY-MM-DD). Defaults to 3 days from now")
    parser.add_argument("--pickup-time",
                       help="Pickup time (HH:MM). Default: read from the site")
    parser.add_argument("--return-time",
                       help="Return time (HH:MM). Default: read from the site")
    parser.add_argument("--headless", action="store_true", 
                       help="Run in headless mode")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ai-first", dest="ai_mode", action="store_const",
                       const="ai_first", default=None,
                       help="Let Ollama drive extraction/navigation (overrides AI_FIRST env)")
    mode.add_argument("--ai-merge", dest="ai_mode", action="store_const",
                       const="ai_merge",
                       help="Run Ollama first, then merge parser/DOM results")
    mode.add_argument("--hybrid", dest="ai_mode", action="store_const",
                       const="hybrid",
                       help="Use deterministic parsers first, Ollama only as fallback")
    
    args = parser.parse_args()
    
    # CLI flag wins over env vars; when no mode flag is passed, keep the env.
    if args.ai_mode == "ai_first":
        agent_core.AI_FIRST = True
        agent_core.AI_MERGE = False
    elif args.ai_mode == "ai_merge":
        agent_core.AI_FIRST = True
        agent_core.AI_MERGE = True
    elif args.ai_mode == "hybrid":
        agent_core.AI_FIRST = False
        agent_core.AI_MERGE = False

    agent_core.AI_PICK_CONFIDENCE = 0.3 if agent_core.AI_FIRST else 0.4
    if agent_core.AI_FIRST and agent_core.AI_MERGE:
        ai_mode_label = "AI-MERGE (Ollama first + parser/DOM merge)"
    elif agent_core.AI_FIRST:
        ai_mode_label = "AI-FIRST (Ollama drives everything)"
    else:
        ai_mode_label = "Hybrid"
    
    pickup_label = args.pickup_date or "from site"
    return_label = args.return_date or "from site"
    pickup_time_label = args.pickup_time or "from site"
    return_time_label = args.return_time or "from site"

    print("="*60)
    print(f"  Universal Car-Rental Scraper")
    print(f"  Site     : https://www.{args.site}.ca")
    print(f"  Location : {args.location}")
    print(f"  Pickup   : {pickup_label} at {pickup_time_label}")
    print(f"  Return   : {return_label} at {return_time_label}")
    print(f"  Headless : {args.headless}")
    print(f"  AI mode  : {ai_mode_label}")
    print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    logs = []
    
    if args.site == "sixt":
        cars = run_sixt(
            location=args.location,
            headless=args.headless,
            logs=logs,
            pickup_date=args.pickup_date,
            return_date=args.return_date,
            pickup_time=args.pickup_time,
            return_time=args.return_time,
        )
        site_name = "Sixt_Canada"
    elif args.site == "enterprise":
        cars = run_enterprise(
            location=args.location,
            headless=args.headless,
            logs=logs,
            pickup_date=args.pickup_date,
            return_date=args.return_date,
            pickup_time=args.pickup_time,
            return_time=args.return_time,
        )
        site_name = "Enterprise_Car_Rental"
    else:
        print(f"Site '{args.site}' not implemented yet")
        return
    
    result = {
        "site": site_name,
        "location": args.location,
        "records": cars,
        "count": len(cars),
        "logs": logs
    }
    
    save_results(result)
    
    print(f"\n Scraping complete! Found {len(cars)} vehicles")
    
    if cars:
        print("\n First 5 vehicles:")
        for i, car in enumerate(cars[:5], 1):
            print(f"  {i}. {car.get('car_name', 'Unknown')}: {car.get('price_per_day', 'N/A')}")


if __name__ == "__main__":
    main()
