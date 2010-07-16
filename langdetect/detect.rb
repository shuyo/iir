#!/usr/bin/ruby -Ku

require 'common.rb'
LD::optparser.parse!(ARGV)

class Detector
  def initialize(filename, alpha=1.0, beta=1.0)
    @n_k, @p_ik, @n = open(filename){|f| Marshal.load(f) }
    @n ||= 3
    @p_ik.default = 0
    @alpha = alpha
    @beta = beta
  end
  def n; @n; end
  def init
    @prob = Hash.new
    LD::LANGLIST.each {|lang| @prob[lang] = 1.0 }
    @maxprob = 0
  end
  def append(x)
    return unless @p_ik.key?(x)
    freq = @p_ik[x]
    sum = 0
    LD::LANGLIST.each do |lang|
      @prob[lang] *= (freq[lang] + @alpha) / (@n_k[lang] + @beta)
      sum += @prob[lang]
    end
    @maxprob = 0
    LD::LANGLIST.each do |lang|
      @prob[lang] /= sum
      @maxprob = @prob[lang] if @maxprob < @prob[lang]
    end
  end
  def maxprob; @maxprob; end
  def problist; @prob.to_a.select{|x| x[1]>0.1}.sort_by{|x| -x[1]}; end
end
detector = Detector.new(LD::opt[:model])

# Database
db = LD::db_connect
ps_select = db.prepare("select id,title,lang,body from news order by lang")
ps_select.execute

count = Hash.new(0)
correct = Hash.new(0)
detected = Hash.new{|h,k| h[k]=Hash.new(0)}
ngram = LD::Ngram.new(detector.n)
while rs = ps_select.fetch
  id, title, lang, body = rs
  text = LD::decode_entity(title + "\n" + body)

  ngram.clear
  detector.init
  text.scan(/./) do |x|
    ngram.append LD::normalize(x)
    ngram.each do |z|
      detector.append z
    end
    break if detector.maxprob > 0.99
  end

  problist = detector.problist
  puts "#{id},#{lang},#{title},#{problist.inspect}"
  count[lang] += 1
  correct[lang] += 1 if problist[0][0] == lang
  detected[lang][problist[0][0]] += 1
end

count.keys.sort.each do |lang|
  rate = (10000.0 * correct[lang] / count[lang]).to_i / 100.0
  list = detected[lang].to_a.sort_by{|x| -x[1]}.map{|x| x.join(':')}.join(',')
  puts "#{lang} #{correct[lang]} / #{count[lang]} (#{rate}) [#{list}]"
end

