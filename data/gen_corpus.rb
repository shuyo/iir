#!/usr/bin/ruby
# ruby gen_corpus.rb ohenry/1444.zip ohenry/1646.zip ohenry/1725.zip ohenry/2777.zip ohenry/2776.zip

require 'pstore'
require "../lib/extract_gutenberg.rb"
require '../lib/infinitive.rb'
INF = Infinitive.new

DEST_DIR = "output"
DEBUG = false

docs = Array.new
terms = Hash.new

doc_id = 0
ARGV.each do |path|
  file = path.dup
	file = $1 if path =~ /\/([^\/]+)$/
	text = nil
	if path =~ /\.zip$/i
	  file.sub!(/\.zip$/i, ".txt")
	  text = `unzip -cq #{path} "*.txt"`
    open("#{DEST_DIR}/#{file}.org", "w"){|f| f.write text}
	else
	  text = open(path){|f| f.read}
	end
  text = Gutenberg.extract(text)
  open("#{DEST_DIR}/#{file}", "w"){|f| f.write text}

  list = text.split(/^[IVX]+\s*\.?$/)[1..-1]
  list = text.split(/^\n{4}$/) if list.size<=1
  list.each do |x|
    next unless x =~ /^(.+)$/
    title = $1

    words = x.scan(/[A-Za-z]+(?:'t)?/)
    next if words.size < 1000

    n = 0
    words.each do |word|
      word = INF.infinitive(word)
      terms[word] ||= Hash.new(0)
      terms[word][doc_id] += 1
      n += 1
    end

    docs[doc_id] = {:title=>title, :n_words=>n}
    doc_id += 1
  end
end

db = PStore.new('corpus')
db.transaction do
  db[:docs] = docs
  db[:terms] = terms
end
