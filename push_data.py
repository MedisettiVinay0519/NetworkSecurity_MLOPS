import os
import sys
import json
from pathlib import Path

import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger

# Load environment variables
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

if not MONGO_DB_URL:
    raise ValueError("MONGO_DB_URL not found in environment variables")

# TLS certificate
ca = certifi.where()


class NetworkDataExtract:
    def __init__(self):
        try:
            self.mongo_client = pymongo.MongoClient(
                MONGO_DB_URL,
                tlsCAFile=ca,
                serverSelectionTimeoutMS=5000
            )
            # Validate connection
            self.mongo_client.admin.command("ping")
            logger.info("Connected to MongoDB successfully")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_convertor(self, file_path: Path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = data.to_dict(orient="records")
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongo_db(self, records, database, collection):
        try:
            db = self.mongo_client[database]
            col = db[collection]
            result = col.insert_many(records)
            return len(result.inserted_ids)
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    try:
        FILE_PATH = Path("Network_Data") / "phisingData.csv"
        DATABASE = "Vinay_Network_Security"
        COLLECTION = "Network_Data"

        networkobj = NetworkDataExtract()
        records = networkobj.csv_to_json_convertor(FILE_PATH)

        no_of_records = networkobj.insert_data_mongo_db(
            records, DATABASE, COLLECTION
        )

        print(f"{no_of_records} records inserted successfully into MongoDB.")

    except Exception as e:
        logger.error(str(e))
