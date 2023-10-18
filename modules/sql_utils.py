from os.path import abspath, dirname, join, exists
import sqlite3


DATABASE_PATH = join(abspath(dirname(__file__)), "SQL", "sqlite.db")


def select_db_row(table, id):
    """select_db_row.

    Selects and prints out a row within a registered sqlite database table.

    Parameters
    ----------
    table : str
        Name of table to select record from.
    id : str
        Id key for required record within table for selection.
    """

    try:
        with sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES) as con:
            cur = con.cursor()
            rows = cur.execute(f"select * from {table} where id={id}")
            for row in rows:
                return row
    except Exception as err:
        print("Database doesn't exist: ", err)


def create_db_table(table):
    """create_db_table.

    Creates a table within sqlite database to store user records.

    Parameters
    ----------
    table : str
        Name of table to create.
    """
    try:
        with sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES) as con:
            cur = con.cursor()
            cur.execute(f"create table {table}(id integer primary key, arr array)")
    except Exception as err:
        print(f"Cannot create table for {table}: ", err)


def establish_sqlite_db(table_name):
    if not exists(DATABASE_PATH):
        sqlite3.connect(DATABASE_PATH.split('/')[-1]).close()
        create_db_table(table_name)

def insert_db_row(table, id, mfcc):
    """insert_db_row.

    Takes required parameters and inserts a record of given id and mfcc dataset into the sqlite database table specified.

    Parameters
    ----------
    table : str
        Name of table to insert record within.
    id : str
        Id key for required record within table for insertion.
    mfcc : numpy.array
        MFCC dataset to be inserted within database records.
    """
    try:
        with sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES) as con:
            cur = con.cursor()
            cur.execute(f"insert into {table}(id, arr) values (?, ?)", (id, mfcc,))
    except Exception as err:
        print("Database doesn't exist: ", err)