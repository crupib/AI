from agent_classes import TripPlannerAgent, TravelSearchAgent, ItineraryFormatterAgent
from travel_database import init_db   # <-- add this

def main():
    init_db()  # <-- make sure the table exists before any tools run

    planner = TripPlannerAgent()
    trip_plan = planner.run()

    searcher = TravelSearchAgent()
    results = searcher.run(trip_plan)

    formatter = ItineraryFormatterAgent()
    markdown_report = formatter.run(results, trip_plan)

    with open("itinerary.md", "w") as f:
        f.write(markdown_report)
    print("\n📝 Itinerary saved to itinerary.md")

if __name__ == "__main__":
    main()