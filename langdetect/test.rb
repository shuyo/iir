#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

require 'common.rb'
require 'detect.rb'

parser= LD::optparser
target_id = nil
alpha = 1.0
parser.on('--id=VAL', Integer, 'target text id') {|v| target_id = v }
parser.on('-a VAL', Float, 'alpha (additive smoothing)') {|v| alpha = v }
parser.parse!(ARGV)

detector = LanguageDetector::Detector.new(LD::model_filename, alpha)

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
  title.sub!(/ - [^\-]+$/, '')
  text = LD::decode_entity(title + "\n" + body)

  ngramer.clear
  detector.init
  text.scan(/./) do |x|
    ngramer.append x
    ngramer.each do |z|
      detector.append z
    end
    break if detector.maxprob > 0.99999
  end

  problist = detector.problist
  puts "#{id},#{lang},#{title},#{problist.inspect}"
  count[lang] += 1
  correct[lang] += 1 if problist[0][0] == lang
  detected[lang][problist[0][0]] += 1
end

sum = correct_sum = 0
count.keys.sort.each do |lang|
  rate = (10000.0 * correct[lang] / count[lang]).to_i / 100.0
  list = detected[lang].to_a.sort_by{|x| -x[1]}.map{|x| x.join(':')}.join(',')
  puts "#{lang} #{correct[lang]} / #{count[lang]} (#{rate}) [#{list}]"
  sum += count[lang]
  correct_sum += correct[lang]
end
puts "total: #{correct_sum} / #{sum} (#{(10000.0 * correct_sum / sum).to_i / 100.0})"

