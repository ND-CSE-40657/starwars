# Simple tokenizer for Chinese and English

use strict;
use warnings;
use open qw(:std :utf8);

while (<>) {
    # Separate punctuation
    s/(\.+|-+|\p{XPosixPunct})/ $1 /g;
    
    # For Chinese, separate into single characters
    s/(\p{Han})/ $1 /g;

    # Undo some splits
    s/n +' +t / n't /g;
    s/ ' +tis / 'tis /g;
    s/ ' +ve / 've /g;
    s/ ' +d / 'd /g;
    s/ ' +s / 's /g;
    s/ ' +m / 'm /g;
    s/ ' +n / 'n /g;
    s/ ' +ll / 'll /g;
    s/ ' +re / 're /g;
    
    # Remove extra spaces
    s/ +/ /g;
    s/^ //g;
    s/ $//g;
    s/\t /\t/g;
    s/ \t/\t/g;
    
    print;
}
