from event.pub_sub import Publisher, Subscriber

publisher = Publisher()
subscriber = Subscriber("Subscriber")
publisher.subscribe(subscriber)
