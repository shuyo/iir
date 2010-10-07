#!/usr/bin/env ruby

traintext = open("Emma.txt") {|f| f.read }
testtext = open("Persuation.txt") {|f| f.read }

count = 0
freq = Hash.new(0)
traintext.scan(/\w+/) do |word|
  freq[word.downcase] += 1
  count += 1
end
count = count.to_f
unigram_MLE = Hash.new
freq.each do |word, f|
  unigram_MLE[word] = f / count
end

def loglikelihood(text, unigram)
  logL = 0.0
  count = 0
  text.scan(/\w+/) do |word|
    w = word.downcase
    if unigram.key?(w)
      logL += Math.log(unigram[w])
      count += 1
    end
  end
  [logL, count]
end

def entropy(text, unigram)
  ent = 0.0
  text.scan(/\w+/) do |word|
    w = word.downcase
    ent -= unigram[w] * Math.log(unigram[w])
  end
  ent
end


puts "MLE:"
logL, count = loglikelihood(traintext, unigram_MLE)
crossent = -logL / count / Math.log(2)
puts "logL of train text(#{count} words) = #{logL}, cross entropy = #{crossent}"
logL, count = loglikelihood(testtext, unigram_MLE)
crossent = -logL / count / Math.log(2)
puts "logL of test text(#{count} words, ignore unseen words) = #{logL}, cross entropy = #{crossent}"


#puts "logL of train text = #{loglikelihood(traintext, unigram_MLE)}, entropy = #{entropy(traintext, unigram_MLE)}"
#puts "logL of test text(ignore unseen words) = #{loglikelihood(testtext, unigram_MLE)}, entropy = #{entropy(testtext, unigram_MLE)}"


