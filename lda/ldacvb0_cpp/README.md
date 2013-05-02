LDA CVB0 in C++
======================


How to Build
------

	git clone git://github.com/shuyo/iir.git
	cd iir/lda/ldacvb0_cpp
	git clone git://github.com/herumi/cybozulib.git
	MSBuild.exe ldacvb0.sln /p:Configuration=Release /p:Platform="Win32"


Usage
------

On cygwin,

	curl http://nltk.googlecode.com/svn/trunk/nltk_data/packages/corpora/brown.zip -O
	unzip brown.zip
	Release/ldacvb0.exe brown/????


Options
------

+ -k : topic size (20)
+ -i : number of learning iteration (100)
+ -a : parameter alpha (0.1)
+ -b : parameter beta (0.01)
+ -n : how many top to print in topic-word distribution (20)
+ -p : use corpus with POS annotation


License
----------

Copyright &copy; 2013 Nakatani Shuyo / Cybozu Labs, Inc

Distributed under the [MIT License][mit].

[MIT]: http://www.opensource.org/licenses/mit-license.php
