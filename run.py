# """
# run.py — CLI entrypoint for the Universal AI Scraper Agent

# Usage examples:
#   # Named preset
#   python run.py --site sixt --location "YYC" --pickup "2025-07-01"
#   python run.py --site enterprise --location "Toronto Pearson Airport"
#   python run.py --site alo_yoga --query "leggings"

#   # Fully custom
#   python run.py --url "https://www.bestbuy.ca" \
#                 --goal "Scrape laptop listings: name, price, brand, rating" \
#                 --query "gaming laptop"

#   # Compare multiple sites
#   python run.py --compare sixt enterprise avis --location "YYC"
# """

# import argparse
# import json
# from agent_core import ScraperAgent, save_results
# from site_configs import SITE_REGISTRY, custom as custom_config


# def build_config_from_args(args) -> dict:
#     if args.site:
#         factory = SITE_REGISTRY.get(args.site)
#         if not factory:
#             raise ValueError(f"Unknown site preset '{args.site}'. Available: {list(SITE_REGISTRY.keys())}")
#         import inspect
#         sig = inspect.signature(factory)
#         kwargs = {}
#         if "location" in sig.parameters and args.location:
#             kwargs["location"] = args.location
#         if "search_query" in sig.parameters and args.query:
#             kwargs["search_query"] = args.query
#         if "pickup_date" in sig.parameters and args.pickup:
#             kwargs["pickup_date"] = args.pickup
#         if "dropoff_date" in sig.parameters and args.dropoff:
#             kwargs["dropoff_date"] = args.dropoff
#         return factory(**kwargs)

#     if args.url and args.goal:
#         params = {}
#         if args.location:
#             params["location"] = args.location
#         if args.query:
#             params["search_query"] = args.query
#         return custom_config(url=args.url, goal=args.goal, **params)

#     raise ValueError("Provide --site <preset> or --url + --goal")


# def main():
#     parser = argparse.ArgumentParser(description="Universal AI Scraper Agent")
#     parser.add_argument("--site", help="Named site preset (sixt, enterprise, avis, alo_yoga, ...)")
#     parser.add_argument("--compare", nargs="+", help="Scrape multiple sites and combine results")
#     parser.add_argument("--url", help="Custom site URL")
#     parser.add_argument("--goal", help="What to scrape (plain English)")
#     parser.add_argument("--location", help="Location / city / airport code")
#     parser.add_argument("--query", help="Search query")
#     parser.add_argument("--pickup", help="Pickup date (YYYY-MM-DD)")
#     parser.add_argument("--dropoff", help="Dropoff date (YYYY-MM-DD)")
#     parser.add_argument("--headless", action="store_true", help="Run browser headless")
#     parser.add_argument("--output", default="./scraper_outputs", help="Output directory")
#     parser.add_argument("--list-sites", action="store_true", help="List available presets")
#     args = parser.parse_args()

#     if args.list_sites:
#         print("Available site presets:")
#         for name in SITE_REGISTRY:
#             print(f"  --site {name}")
#         return

#     if args.compare:
#         # Multi-site comparison mode
#         all_records = []
#         for site_name in args.compare:
#             args.site = site_name
#             config = build_config_from_args(args)
#             print(f"\n{'='*60}")
#             print(f"🌐 Scraping: {site_name.upper()}")
#             print(f"{'='*60}")
#             agent = ScraperAgent(config, headless=args.headless)
#             result = agent.run()
#             for r in result["records"]:
#                 r["__source_site"] = site_name
#             all_records.extend(result["records"])
#             print(f"✅ {site_name}: {result['count']} records")

#         # Save combined
#         import os, re, pandas as pd
#         from datetime import datetime
#         os.makedirs(args.output, exist_ok=True)
#         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#         sites_slug = "_vs_".join(args.compare)[:40]
#         csv_path = os.path.join(args.output, f"comparison_{sites_slug}_{ts}.csv")
#         json_path = csv_path.replace(".csv", ".json")
#         if all_records:
#             pd.DataFrame(all_records).to_csv(csv_path, index=False)
#             with open(json_path, "w") as f:
#                 json.dump(all_records, f, indent=2)
#             print(f"\n📊 Combined: {len(all_records)} records → {csv_path}")
#         return

#     # Single site
#     config = build_config_from_args(args)
#     agent = ScraperAgent(config, headless=args.headless)
#     result = agent.run()
#     paths = save_results(result, args.output)
#     print(f"\n🎉 Done! {result['count']} records")
#     print(f"   CSV  → {paths.get('csv', 'N/A')}")
#     print(f"   JSON → {paths.get('json', 'N/A')}")
#     print(f"   Logs → {paths.get('logs', 'N/A')}")


# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# """
# run.py — Universal Car-Rental Scraper CLI
# ==========================================

# Usage examples
# --------------
#   python3 run.py --site sixt      --location "Las Vegas"
#   python3 run.py --site enterprise --location "Calgary"
#   python3 run.py --site sixt      --location YYC
#   python3 run.py --site enterprise --location "Toronto Pearson Airport"
#   python3 run.py --site sixt      --location "New York" --headless
#   python3 run.py --url "https://www.avis.com" --location "Vancouver" --goal "Find all rental cars"

# Output
# ------
#   ./scraper_outputs/<site>/<location>/<MST_timestamp>.csv
#   ./scraper_outputs/<site>/<location>/<MST_timestamp>.json
#   ./scraper_outputs/<site>/<location>/<MST_timestamp>.log
# """

# import argparse
# import sys

# from agent_core import ScraperAgent, save_results, now_mst

# SITE_URLS = {
#     "sixt": "https://www.sixt.ca",
#     "enterprise": "https://www.enterprise.com/en/car-rental/locations.html",
# }


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Universal AI Car-Rental Scraper",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog=__doc__,
#     )

#     group = parser.add_mutually_exclusive_group(required=True)
#     group.add_argument(
#         "--site",
#         choices=list(SITE_URLS.keys()),
#         help="Shortcut for a known site (sixt | enterprise)",
#     )
#     group.add_argument(
#         "--url",
#         help="Full URL for any other site (uses generic AI planner)",
#     )

#     parser.add_argument(
#         "--location",
#         required=True,
#         help='City, airport name, or IATA code (e.g. "Calgary", YYC, "Las Vegas")',
#     )
#     parser.add_argument(
#         "--goal",
#         default="Extract all available car rental listings with prices",
#         help="What to scrape (used by the generic AI planner for unknown sites)",
#     )
#     parser.add_argument(
#         "--headless",
#         action="store_true",
#         default=False,
#         help="Run the browser in headless mode (no visible window)",
#     )
#     parser.add_argument(
#         "--output-dir",
#         default="scraper_outputs",
#         help="Root output directory (default: ./scraper_outputs)",
#     )

#     return parser.parse_args()


# def main():
#     args = parse_args()

#     url = SITE_URLS.get(args.site, "") if args.site else args.url

#     site_config = {
#         "url": url,
#         "goal": args.goal,
#         "search_params": {
#             "location": args.location,
#         },
#     }

#     print(f"\n{'='*60}")
#     print(f"  Universal Car-Rental Scraper")
#     print(f"  Site     : {url}")
#     print(f"  Location : {args.location}")
#     print(f"  Headless : {args.headless}")
#     print(f"  Time (MST): {now_mst().strftime('%Y-%m-%d %H:%M:%S %Z')}")
#     print(f"{'='*60}\n")

#     agent = ScraperAgent(site_config, headless=args.headless)
#     result = agent.run()

#     if result["count"] == 0:
#         print("\n⚠  No records were scraped.")
#         print("   Check the debug screenshots in /tmp/ for clues.")
#         sys.exit(1)

#     print(f"\n Done! {result['count']} records scraped.")
#     paths = save_results(result, output_base=args.output_dir)

#     print("\nFiles saved:")
#     for kind, path in paths.items():
#         print(f"  {kind:<5} → {path}")
#     print()


# if __name__ == "__main__":
#     main()



































#!/usr/bin/env python3
"""
Universal Car Rental Scraper - CLI Interface
"""

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


















# #!/usr/bin/env python3
# """
# Universal Car Rental Scraper - CLI Interface
# """

# import argparse
# from datetime import datetime, timedelta
# from agent_core import run_sixt, run_enterprise, save_results


# def main():
#     parser = argparse.ArgumentParser(description="Universal Car Rental Scraper")
#     parser.add_argument("--site", required=True, choices=["sixt", "enterprise"], 
#                        help="Rental site to scrape")
#     parser.add_argument("--location", required=True, 
#                        help="Pickup location (e.g., 'Toronto Airport' or 'YUL')")
#     parser.add_argument("--pickup-date", 
#                        help="Pickup date (YYYY-MM-DD). Defaults to tomorrow")
#     parser.add_argument("--return-date", 
#                        help="Return date (YYYY-MM-DD). Defaults to 3 days from now")
#     parser.add_argument("--headless", action="store_true", 
#                        help="Run in headless mode")
    
#     args = parser.parse_args()
    
#     # Set default dates
#     if not args.pickup_date:
#         args.pickup_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
#     if not args.return_date:
#         args.return_date = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    
#     print("="*60)
#     print(f"  Universal Car Rental Scraper")
#     print(f"  Site     : {args.site.upper()}")
#     print(f"  Location : {args.location}")
#     print(f"  Pickup   : {args.pickup_date}")
#     print(f"  Return   : {args.return_date}")
#     print(f"  Headless : {args.headless}")
#     print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print("="*60)
#     print()
    
#     logs = []
    
#     if args.site == "sixt":
#         vehicles = run_sixt(
#             location=args.location,
#             headless=args.headless,
#             logs=logs
#         )
#         site_name = "SIXT"
#     else:  # enterprise
#         vehicles = run_enterprise(
#             location=args.location,
#             headless=args.headless,
#             logs=logs
#         )
#         site_name = "Enterprise"
    
#     result = {
#         "site": site_name,
#         "location": args.location,
#         "records": vehicles,
#         "count": len(vehicles),
#         "logs": logs
#     }
    
#     save_results(result)
    
#     print(f"\n{'='*60}")
#     print(f"SCRAPING COMPLETE!")
#     print(f"📊 Found {len(vehicles)} vehicles from {site_name}")
#     print(f"{'='*60}")
    
#     if vehicles:
#         print("\n📋 Vehicles found:")
#         for i, v in enumerate(vehicles[:10], 1):
#             name = v.get('car_name', 'Unknown')
#             price = v.get('price_per_day', 'N/A')
#             print(f"  {i}. {name} - {price}")


# if __name__ == "__main__":
#     main()