#!/usr/bin/ruby
# generate libsvm-format data
# gen_libsvm.rb [positive] [negative]
if ARGV.size<2
  puts "gen_libsvm.rb [positive] [negative]"
  exit 1
end

require "../lib/extract_gutenberg.rb"
require '../lib/infinitive.rb'
INF = Infinitive.new

def extract_words(path, terms)
	file = $1 if path =~ /\/([^\/]+)$/
	text = if path =~ /\.zip$/i
	  file.sub!(/\.zip$/i, ".txt")
	  `unzip -cq #{path} "*.txt"`
	else
	  open(path){|f| f.read}
	end
  text = Gutenberg.extract(text)

  list = text.split(/^[IVX]+\s*\.?$/)[1..-1]
  list = text.split(/^\n{4}$/) if list.size<=1
  docs = Array.new
  list.each do |x|
    next unless x =~ /^(.+)$/
    title = $1

    words = x.scan(/[A-Za-z]+(?:'t)?/)
    next if words.size < 1000

    n = 0
    doc_id = docs.size
    words.each do |word|
      word = INF.infinitive(word)
      terms[word] ||= Hash.new(0)
      terms[word][doc_id] += 1
      n += 1
    end
    docs << {:title=>title, :n_words=>n}
  end
  docs
end

terms = Hash.new
positive_docs = extract_words(ARGV[0], terms)
negative_docs = extract_words(ARGV[1], terms)

