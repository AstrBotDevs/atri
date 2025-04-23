import aiosqlite
import os


class DocumentStorage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self.sqlite_init_path = os.path.join(
            os.path.dirname(__file__), "sqlite_init.sql"
        )

    async def initialize(self):
        """Initialize the SQLite database and create the documents table if it doesn't exist."""
        if not os.path.exists(self.db_path):
            await self.connect()
            async with self.connection.cursor() as cursor:
                with open(self.sqlite_init_path, "r") as f:
                    sql_script = f.read()
                await cursor.executescript(sql_script)
            await self.connection.commit()
        else:
            await self.connect()

    async def connect(self):
        """Connect to the SQLite database."""
        self.connection = await aiosqlite.connect(self.db_path)

    async def get_document(self, id: int):
        """Retrieve a document by its ID.

        Args:
            id (int): The ID to retrieve.

        Returns:
            dict: The document data.
        """
        async with self.connection.cursor() as cursor:
            await cursor.execute("SELECT * FROM documents WHERE id = ?", (id,))
            row = await cursor.fetchone()
            if row:
                return await self.tuple_to_dict(row)
            else:
                return None

    async def get_document_by_doc_id(self, doc_id: str):
        """Retrieve a document by its doc_id.

        Args:
            doc_id (str): The doc_id of the document to retrieve.

        Returns:
            dict: The document data.
        """
        async with self.connection.cursor() as cursor:
            await cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            row = await cursor.fetchone()
            if row:
                return await self.tuple_to_dict(row)
            else:
                return None

    async def tuple_to_dict(self, row):
        """Convert a tuple to a dictionary.

        Args:
            row (tuple): The row to convert.

        Returns:
            dict: The converted dictionary.
        """
        return {
            "id": row[0],
            "doc_id": row[1],
            "text": row[2],
            "meta": row[3],
            "created_at": row[4],
            "updated_at": row[5],
        }

    async def close(self):
        """Close the connection to the SQLite database."""
        if self.connection:
            await self.connection.close()
            self.connection = None
