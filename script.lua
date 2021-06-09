previous_enemies = 0
previous_level = 0
function correct_enemies()
  if done_check() then
    return -10
  end
  if data.enemies > previous_enemies then
    local delta = data.enemies - previous_enemies
    previous_enemies = data.enemies
    return delta
  else
    return 0
  end
  
end

function done_check()
  if data.lives == 0 then
    return true
  end
  return false
end

function next_level()
if data.enemies == 0 then
    local up = previous_level + 1
    previous_level = data.level
    return up
  else
    return 0
  end