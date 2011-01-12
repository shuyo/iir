#!/usr/bin/env ruby

testtext = open(ARGV[0]) {|f| f.read }
traintext = open(ARGV[1]) {|f| f.read }

count = 0
freq = Hash.new(0)
freq2 = Hash.new(0)
pre = nil
traintext.scan(/\w+/) do |word|
  word.downcase!
  freq[word] += 1
  freq2[pre+" "+word] += 1 if pre
  pre = word
  count += 1
end

class MLE
  def initialize(freq, freq2, count)
    @prob = Hash.new(1)
    @prob2 = Hash.new(1)
    count = count.to_f
    freq.each do |word, f|
      @prob[word] = f / count
    end
    freq2.each do |word, f|
      @prob2[word] = f / (count-1)
    end
  end
  def set_target(text);end
  def key?(word);@prob.key?(word);end
  def [](word);@prob[word];end
  def bigram(word, word2);@prob2[word+" "+word2]/@prob[word];end
end
class Laplace
  def initialize(freq, freq2, count, lambda=1.0)
    @freq = freq.clone
    @freq2 = freq2.clone
    @count = count
    @lambda = lambda
  end
  def set_target(text)
    @voca = @freq.clone
    text.scan(/\w+/) do |word|
      @voca[word] = 0 unless @voca.key?(word)
    end
    @V = @voca.size
  end
  def key?(word);@voca.key?(word);end
  def [](word);(@voca[word]+@lambda)/(@count+@V*@lambda);end
  def bigram(word, word2)
    (@freq2[word+" "+word2]+@lambda)/(@count-1+@V*@V*@lambda)/self[word]
  end
end

model_MLE = MLE.new(freq, freq2, count)
model_Laplace = Laplace.new(freq, freq2, count)
model_Laplace.set_target(testtext)
model_ELE = Laplace.new(freq, freq2, count, 0.5)
model_ELE.set_target(testtext)

def loglikelihood(text, model)
  logL = 0.0  # unigram
  logL2 = 0.0 # bigram
  count = 0
  pre = nil
  text.scan(/\w+/) do |word|
    word.downcase!
    if model.key?(word)
      logL += Math.log(model[word])
      count += 1
      if pre
        logL2 += Math.log(model.bigram(pre, word))
      else
        logL2 += Math.log(model[word])
      end
      pre = word
    end
  end
  [logL, logL2, count]
end

def entropy(text, model)
  ent = 0.0
  text.scan(/\w+/) do |word|
    w = word.downcase
    ent -= model[w] * Math.log(model[w])
  end
  ent
end


puts "MLE:"
logL, logL2, count = loglikelihood(traintext, model_MLE)
crossent = -logL / count / Math.log(2)
crossent2 = -logL2 / count / Math.log(2)
puts "train text(#{count} words)"
puts "logL of unigrams = #{logL}, cross entropy = #{crossent}"
puts "logL of bigrams = #{logL2}, cross entropy = #{crossent2}"
#logL, count = loglikelihood(testtext, model_MLE)
#crossent = -logL / count / Math.log(2)
#puts "logL of test text(#{count} words, ignore unseen words) = #{logL}, cross entropy = #{crossent}"

puts "Laplace:"
logL, logL2, count = loglikelihood(traintext, model_Laplace)
crossent = -logL / count / Math.log(2)
crossent2 = -logL2 / count / Math.log(2)
puts "logL of train text(#{count} words) = #{logL}, cross entropy = #{crossent}"
puts "logL2 of train text(#{count} words) = #{logL2}, cross entropy 2 = #{crossent2}"
logL, logL2, count = loglikelihood(testtext, model_Laplace)
crossent = -logL / count / Math.log(2)
crossent2 = -logL2 / count / Math.log(2)
puts "logL of test text(#{count} words) = #{logL}, cross entropy = #{crossent}"
puts "logL2 of test text(#{count} words) = #{logL2}, cross entropy 2 = #{crossent2}"

puts "ELE:"
logL, logL2, count = loglikelihood(traintext, model_ELE)
crossent = -logL / count / Math.log(2)
crossent2 = -logL2 / count / Math.log(2)
puts "logL of train text(#{count} words) = #{logL}, cross entropy = #{crossent}"
puts "logL2 of train text(#{count} words) = #{logL2}, cross entropy 2 = #{crossent2}"
logL, logL2, count = loglikelihood(testtext, model_ELE)
crossent = -logL / count / Math.log(2)
crossent2 = -logL2 / count / Math.log(2)
puts "logL of test text(#{count} words) = #{logL}, cross entropy = #{crossent}"
puts "logL2 of test text(#{count} words) = #{logL2}, cross entropy 2 = #{crossent2}"


#puts "logL of train text = #{loglikelihood(traintext, model_MLE)}, entropy = #{entropy(traintext, model_MLE)}"
#puts "logL of test text(ignore unseen words) = #{loglikelihood(testtext, model_MLE)}, entropy = #{entropy(testtext, model_MLE)}"


