#!/bin/bash

xattr -d -r com.apple.quarantine __main__.dist
__main__.dist/__main__ server $@