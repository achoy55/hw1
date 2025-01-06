import redis
import pandas as pd
import pyarrow as pa
from io import BytesIO

class RedisClient:
    __r = None
    def __init__(self, host='localhost', port=6379, db=0, username=None, password=None):
        self.host=host
        self.port=port
        self.db=db
        self.username=username
        self.password=password,
        self.decode_responses=False
        
        self.__r = self._connect(db=1, username='usr_redis', password='usr_pwd')

    def _connect(self, host='localhost', port=6379, db=0, username=None, password=None):
        return redis.Redis(host=host, port=port, db=db, username=username, password=password, decode_responses=False)

    def disconnect(self):
        self.__r.close()

    def flusgdb(self):
        self.__r.flushdb()

    def set_key(self, key, df):
        data = df.to_parquet(compression='gzip', index=True)
        self.__r.set(key, data)

    def get_key(self, key):
        pq_file = BytesIO(self.__r.get(key))
        if pq_file.getbuffer().nbytes == 0:
            return None
        return pd.read_parquet(pq_file)

    def test_connection(self):
        try:
            info = self.__r.info()
            print(info['redis_version'])
            response = self.__r.ping()
            if response:
                print("Connected to Redis")
            else:
                print("Failed connect to Redis")
        except redis.exceptions.RedisError as e:
            print(f"Error: {e}")



## Testing
if __name__ == "__main__":
    r = RedisClient(db=1, username='usr_redis', password='usr_pwd')
    r.test_connection()
    # df = pd.DataFrame({'one': [-1, 3.2, 2.5],
    #                'two': ['foo', 'bar', 'baz'],
    #                'three': [True, False, True]},
    #                index=list('abc'))
    # r.set_key('df', df)
    # print(r.get_key('df'))
    