import os

from qdrant_client import QdrantClient


class QdrantDBFactory(object):
    client: QdrantClient = QdrantClient(
        "https://d638b054-362e-411d-90e3-88e0f9e35b59.eu-central-1-0.aws.cloud.qdrant.io",
        prefer_grpc=True,
        api_key=os.environ["QDRANT_API_KEY"],
    )

    def get_client(self) -> QdrantClient:
        return self.client
