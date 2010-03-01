#!/usr/bin/ruby -KN
# ./bs.rb [corpus file] [query ...]

require '../lib/infinitive.rb'
INF = Infinitive.new

docs = Array.new
words = Hash.new{|h,k| h[k]=Hash.new }
worddocs = Hash.new{|h,k| h[k]=Hash.new }
open(ARGV[0]) do |f|
  while line = f.gets
    vec = Hash.new(0)
    doc_id = docs.length
    line.scan(/[A-Za-z]+/) do |word|
      infword = INF.infinitive(word)
      vec[infword] = 1
      words[infword][word] = 1
      worddocs[infword][doc_id] = 1
    end
    docs << vec if vec.size > 0
  end
end

class BayesianSet
  C = 2.0
  def initialize(docs, words, worddocs)
    @docs = docs
    @words = words
    @worddocs = worddocs
    @alpha = docs.map{|vec| C * vec.size / words.length }
    @beta = @alpha.map{|a| C - a }
    puts "# of words = #{words.size}, # of docs = #{docs.length}"
  end

  def search(query)
    query = query.map{|x| INF.infinitive(x)}.uniq
    n = query.length
    alpha_tild = Array.new # ln(alpha~/alpha)
    beta_tild = Array.new  # ln(beta~/beta)
    @alpha.each_with_index do |a, i|
      s = query.select{|w| @docs[i].key?(w) }.length
      alpha_tild << Math.log(1 + s / a)
      beta_tild << Math.log(1 + (n - s) / @beta[i])
    end

    @worddocs.map do |w, docs|
      score = 0
=begin
      # method of original paper
      @docs.each_with_index do |vec, j|
        score += Math.log(@alpha[j]+@beta[j])-Math.log(@alpha[j]+@beta[j]+n)
        score += if vec.key?(w) then alpha_tild[j] else beta_tild[j] end
      end
=end
      # simple & fast
      docs.each do |j, dummy|
        score += alpha_tild[j] - beta_tild[j]
      end

      [w, score]
    end.sort_by{|x| -x[1]}[0..9].each do |w, score|
      puts "#{w}: #{score} (#{@words[w].keys.join(',')})"
    end
  end
end

bs = BayesianSet.new(docs, words, worddocs)
if ARGV.length > 2
  bs.search(ARGV[1..-1])
else
  while input = $stdin.gets
    bs.search(input.split)
  end
end

