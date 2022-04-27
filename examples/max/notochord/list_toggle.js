outlets = 1;
setinletassist(0,"note");
setoutletassist(0,"list of pitches");

arr = []

function msg_int(a)
{	
	if (!contains(arr, a)) {
		arr.push(a)
	} else {
		arr = arr.filter(function(e) {return e !== a})
	}
	
	outlet(0, arr)
}

function contains(a, obj) {
    var i = a.length;
    while (i--) {
       if (a[i] === obj) {
           return true;
       }
    }
    return false;
}
