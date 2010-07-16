#!/usr/bin/ruby -Ku

require 'common.rb'
require 'detect.rb'
parser= LD::optparser
target_id = nil
parser.on('--id=VAL', Integer, 'target text id') {|v| target_id = v }
parser.parse!(ARGV)

detector = LanguageDetector::Detector.new(LD::model_filename, 100.0, 100.0)

# Database
db = LD::db_connect
ps_select = if target_id
  detector.debug_on
  db.prepare("select id,title,lang,body from news where id = ?").execute target_id
else
  db.prepare("select id,title,lang,body from news order by lang").execute
end

count = Hash.new(0)
correct = Hash.new(0)
detected = Hash.new{|h,k| h[k]=Hash.new(0)}
ngramer = detector.ngramer
while rs = ps_select.fetch
  id, title, lang, body = rs
  text = LD::decode_entity(title + "\n" + body)

  ngramer.clear
  detector.init
  text.scan(/./) do |x|
    ngramer.append x
    ngramer.each do |z|
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

