#!/usr/bin/ruby

require 'rubygems'
require 'stemmer'
require 'linguistics'
Linguistics::use( :en )

class Infinitive
  def initialize(inflist_file = nil, wordbook_file = nil)
    dir = caller(0)[0].sub(/\/[^\/]*:\d+:.+$/,"")
    inflist_file = "#{dir}/inflist.txt" unless inflist_file
    wordbook_file = "#{dir}/wordbook.txt" unless wordbook_file

    @inflist = Hash.new
    open(inflist_file) do |f|
      while line = f.gets
        @inflist[$1]=$2 if line =~ /^(.+)\t(.+)\n/
      end
    end
    @infcache = Hash.new

    @wordbook = Hash.new
    open(wordbook_file) do |f|
      while line = f.gets
        @wordbook[line.chomp.stem.downcase]=0 if line !~ /^\s+$|^\s*#/
      end
    end
  end

  def inf(src)
    return @infcache[src] if @infcache.key?(src)
    st = @inflist[src] || src.en.infinitive
    @infcache[src] = (if st == "" then src else st end).stem
  end

  def infinitive(word)
    st = word2 = word.downcase
    st = word2.stem
    if @wordbook.key?(st)
      st
    else
      inf(word2) || st
    end
  end
end

