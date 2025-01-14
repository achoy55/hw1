import queue

class Publisher:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.subscribers = []
 
    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
 
    def publish(self, message):
        self.message_queue.put(message)
        for subscriber in self.subscribers:
            subscriber.receive(message)

class Subscriber:
    def __init__(self, name):
        self.name = name
 
    def receive(self, message):
        invoke_func = message['invoke_func']
        params = message['params']
        # try:
        invoke_func(params)
        # except Exception as e:
        #     print(f'Error invoke function; {invoke_func}: {e}')



def test():
    print('Ploting')

def test2():
    print('Data proc')

if __name__ == "__main__":
    pass
    # publisher = Publisher()
    # subscriber_1 = Subscriber("Subscriber", test, test2)
    # publisher.subscribe(subscriber_1)
    # publisher.publish("data")