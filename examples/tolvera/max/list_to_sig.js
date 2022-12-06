//outlets = 1;
setinletassist(0,"list");
//setoutletassist(0,"buffer");

var buf = new Buffer("lenia")

function list()
{
	arr = arrayfromargs(arguments)
	buf.poke(1,0,arr)
}
