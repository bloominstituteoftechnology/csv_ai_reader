from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()


csv = "./datasets/yelp_reviews.csv"
llm = ChatOpenAI(temperature=0.5)
yelp_agent = create_csv_agent(llm, csv, allow_dangerous_code=True)


def scrape_site():
    scraper = input("What website do you want to scrape? ")
    loader = WebBaseLoader(scraper)
    docs = loader.load()
    web_content = docs[0].page_content
    return web_content


def extract_review(wc):
    first_review = llm.invoke(
        f"Extract the first review from the following url: {wc}"
    ).content
    return first_review


site_extract = scrape_site()
first_review = extract_review(site_extract)

yelp_agent.invoke(
    f"Take this review: {first_review}. Add the review to the dataframe as a new row. Create or update a new file called updated_reviews.csv"
)
