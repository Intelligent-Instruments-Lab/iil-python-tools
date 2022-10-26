b = NetAddr("localhost", 8888)

b.sendMsg(\feed, Process.elapsedTime.sin)

b.sendMsg(\query)

b.sendMsg(\reset)

(
OSCdef(\predictor, { arg msg;
	msg.postln;

}, "/query_return",)
)