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

REG_TITLE = /^([A-Z][A-Z\-', ]+)$/
class Analyzer
  def initialize
    @terms = Hash.new
    @docs = Array.new
  end
  attr_reader :docs, :terms
  def extract_words(path, sign)
    file = path.dup
  	file = $1 if path =~ /\/([^\/]+)$/
  	text = if path =~ /\.zip$/i
  	  file.sub!(/\.zip$/i, ".txt")
  	  `unzip -cq #{path} "*.txt"`
  	else
  	  open(path){|f| f.read}
  	end
    text = Gutenberg.extract(text)

    list = text.split(REG_TITLE)

    title = nil
    list.each do |x|
      if x =~ REG_TITLE
        title = x
        next
      end
      words = x.scan(/[A-Za-z]+(?:'t)?/)
      next if words.size < 1000

      while words.size >= 100
        subwords = words.slice!(0, 100)
        n = 0
        doc_id = @docs.size
        subwords.each do |word|
          word = INF.infinitive(word)
          @terms[word] ||= Hash.new(0)
          @terms[word][doc_id] += 1
          n += 1
        end
        @docs << {:title=>title, :n_words=>n, :sign=>sign}
      end
    end
  end
end

ana = Analyzer.new
ana.extract_words ARGV[0], "+1"
ana.extract_words ARGV[1], "-1"

words = ana.terms.keys
ana.docs.each_with_index do |doc, doc_id|
  buf = [doc[:sign]]
  words.each_with_index do |word, word_id|
    if ana.terms[word]
      freq = ana.terms[word][doc_id]
      buf << "#{word_id}:#{freq}" if freq>0
    end
  end
  puts buf.join(' ')
end

