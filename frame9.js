function xdsAction()
{
   this.xue.gotoAndStop(int(this.life / this.xxx * 100));
   if(this._currentframe == 146)
   {
      bsxlh.push(this.number);
      estan.unit--;
      this.removeMovieClip();
   }
   if(this.attackzl == 3 && this._currentframe < 120 && this._x < 160)
   {
      this.attackzl = 0;
   }
   if(this.life <= 0 && this._currentframe < 120)
   {
      this.gotoAndPlay(120);
      zczk2.text = zczk.text;
      zczk.text = "- A monk of Eastan died .";
   }
   if(this.attackzl == 1 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 0;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && people[i].jlife == 1 && people[i].life < people[i].xxx && i < 16)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
               break;
            }
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && i >= 16)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            if(i == 29 && this.goalxs == 0)
            {
               this.goal = ecastle;
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(Math.sqrt(Math.pow(this.goal._x - this._x,2) + Math.pow(this.goal._y - this._y,2)) < 150 && this._currentframe < 28)
      {
         if(this.goal !== ecastle && this.goal !== estan)
         {
            changlayer(this,this.goal);
         }
         this.gotoAndPlay(51);
      }
      if(this.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndPlay(120);
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.goal._currentframe < 120 && this._currentframe > 28 && Math.sqrt(Math.pow(this.goal._x - this._x,2) + Math.pow(this.goal._y - this._y,2)) < 150)
      {
         k = 1;
         while(k < 15)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 75)
         {
            if(this.goal != ecastle)
            {
               if(this.goal.life > this.goal.xxx && this._currentframe < 120)
               {
                  this.gotoAndStop(1);
               }
               if(this.goal.jlife == 1)
               {
                  xdsxg.attachMovie("zly",xdsxs,xdsxs);
                  xdsxg[xdsxs]._y = this.goal._y - 80;
                  xdsxg[xdsxs]._x = this.goal._x;
                  this.goal.life = this.goal.life + this.cure;
               }
               if(this.goal.jlife == 0 or this.goal.jlife == 2)
               {
                  xdsxg.attachMovie("jsgj",xdsxs,xdsxs);
                  xdsxg[xdsxs]._y = this.goal._y - Math.random() * 30;
                  xdsxg[xdsxs]._x = this.goal._x;
                  this.goal.life = this.goal.life - (7 + Math.random() * 7);
               }
               xdsxs++;
               xdsxs = xdsxs % 30;
            }
            if(this.goal == ecastle)
            {
               if(this.goal.life <= 0 && this._currentframe < 120)
               {
                  this.gotoAndStop(1);
               }
               xdsxg.attachMovie("jsgj",xdsxs,xdsxs);
               xdsxg[xdsxs]._y = this.goal._y + Math.random() * 20 - 10;
               xdsxg[xdsxs]._x = this.goal._x + Math.random() * 40 + 20;
               ecastle.life = ecastle.life - (7 + Math.random() * 7);
               xdsxs++;
               xdsxs = xdsxs % 30;
            }
         }
         else
         {
            this.goalxs = 0;
            this.goaljl = 650;
            i = 0;
            while(i < 30)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && people[i].jlife == 1 && people[i].life < people[i].xxx && i < 16)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
                  break;
               }
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && i >= 16)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
               }
               if(i == 29 && this.goalxs == 0)
               {
                  this.goal = ecastle;
               }
               i++;
            }
         }
      }
      else
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 0;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && people[i].jlife == 1 && people[i].life < people[i].xxx && i < 16)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
               break;
            }
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && i >= 16)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            if(i == 29 && this.goalxs == 0)
            {
               this.goal = ecastle;
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         if(this.goal._x > 320)
         {
            this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 100 - this._x));
            if(this.goal._x - 100 - this._x > 0)
            {
               this.dx = Math.cos(this.angle) * this.speed;
               this.dy = Math.sin(this.angle) * this.speed;
            }
            if(this.goal._x - 100 - this._x == 0)
            {
               if(Math.abs(this.goal._y - this._y) >= 180)
               {
                  if(this.goal._y - this._y > 0)
                  {
                     this.dx = 0;
                     this.dy = 1;
                  }
                  if(this.goal._y - this._y < 0)
                  {
                     this.dx = 0;
                     this.dy = -1;
                  }
               }
            }
            if(this.goal._x - 100 - this._x < 0)
            {
               this.dx = (- Math.cos(this.angle)) * this.speed;
               this.dy = (- Math.sin(this.angle)) * this.speed;
            }
            this._x = this._x + this.dx;
            this._y = this._y + this.dy;
         }
      }
      if(this.goal._x <= 320 && Math.sqrt(Math.pow(this.goal._x - this._x,2) + Math.pow(this.goal._y - this._y,2)) >= 150)
      {
         if(Math.abs(this.goal._y - this._y) > 4)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         if(Math.abs(this.goal._y - this._y) <= 4)
         {
            if(this.goal._x - this._x > 0)
            {
               this._x = this._x + 1;
            }
            else
            {
               this._x = this._x - 1;
            }
         }
      }
   }
   if(this.attackzl == 2 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 120;
         i = 0;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && people[i].jlife == 1 && people[i].life < people[i].xxx && i < 16)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
               break;
            }
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && i >= 16)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            if(i == 29 && this.goalxs == 0)
            {
               this.attackzl = 0;
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(Math.sqrt(Math.pow(this.goal._x - this._x,2) + Math.pow(this.goal._y - this._y,2)) < 150 && this._currentframe < 28)
      {
         if(this.goal !== ecastle && this.goal !== estan)
         {
            changlayer(this,this.goal);
         }
         this.gotoAndPlay(51);
      }
      if(this.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndPlay(120);
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.goal._currentframe < 120 && this._currentframe > 28 && Math.sqrt(Math.pow(this.goal._x - this._x,2) + Math.pow(this.goal._y - this._y,2)) < 150)
      {
         k = 1;
         while(k < 15)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 75)
         {
            if(this.goal != ecastle)
            {
               if(this.goal.life > this.goal.xxx && this._currentframe < 120)
               {
                  this.gotoAndStop(1);
               }
               if(this.goal.jlife == 1)
               {
                  xdsxg.attachMovie("zly",xdsxs,xdsxs);
                  xdsxg[xdsxs]._y = this.goal._y - 80;
                  xdsxg[xdsxs]._x = this.goal._x;
                  this.goal.life = this.goal.life + this.cure;
               }
               if(this.goal.jlife == 0 or this.goal.jlife == 2)
               {
                  xdsxg.attachMovie("jsgj",xdsxs,xdsxs);
                  xdsxg[xdsxs]._y = this.goal._y - Math.random() * 30;
                  xdsxg[xdsxs]._x = this.goal._x;
                  this.goal.life = this.goal.life - (7 + Math.random() * 7);
               }
               xdsxs++;
               xdsxs = xdsxs % 30;
            }
            if(this.goal == ecastle)
            {
               if(this.goal.life <= 0 && this._currentframe < 120)
               {
                  this.gotoAndStop(1);
               }
               xdsxg.attachMovie("jsgj",xdsxs,xdsxs);
               xdsxg[xdsxs]._y = this.goal._y + Math.random() * 20 - 10;
               xdsxg[xdsxs]._x = this.goal._x + Math.random() * 40 + 20;
               ecastle.life = ecastle.life - (7 + Math.random() * 7);
               xdsxs++;
               xdsxs = xdsxs % 30;
            }
         }
         else
         {
            this.goalxs = 0;
            this.goaljl = 120;
            i = 0;
            while(i < 30)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && people[i].jlife == 1 && people[i].life < people[i].xxx && i < 16)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
                  break;
               }
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && i >= 16)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
               }
               if(i == 29 && this.goalxs == 0)
               {
                  this.attackzl = 0;
               }
               i++;
            }
         }
      }
      else
      {
         this.goalxs = 0;
         this.goaljl = 120;
         i = 0;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && people[i].jlife == 1 && people[i].life < people[i].xxx && i < 16)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
               break;
            }
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && i >= 16)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            if(i == 29 && this.goalxs == 0)
            {
               this.attackzl = 0;
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         if(this.goal._x > 320)
         {
            this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 100 - this._x));
            if(this.goal._x - 100 - this._x > 0)
            {
               this.dx = Math.cos(this.angle) * this.speed;
               this.dy = Math.sin(this.angle) * this.speed;
            }
            if(this.goal._x - 100 - this._x == 0)
            {
               if(Math.abs(this.goal._y - this._y) >= 180)
               {
                  if(this.goal._y - this._y > 0)
                  {
                     this.dx = 0;
                     this.dy = 1;
                  }
                  if(this.goal._y - this._y < 0)
                  {
                     this.dx = 0;
                     this.dy = -1;
                  }
               }
            }
            if(this.goal._x - 100 - this._x < 0)
            {
               this.dx = (- Math.cos(this.angle)) * this.speed;
               this.dy = (- Math.sin(this.angle)) * this.speed;
            }
            this._x = this._x + this.dx;
            this._y = this._y + this.dy;
         }
      }
      if(this.goal._x <= 320 && Math.sqrt(Math.pow(this.goal._x - this._x,2) + Math.pow(this.goal._y - this._y,2)) >= 150)
      {
         if(Math.abs(this.goal._y - this._y) > 4)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         if(Math.abs(this.goal._y - this._y) <= 4)
         {
            if(this.goal._x - this._x > 0)
            {
               this._x = this._x + 1;
            }
            else
            {
               this._x = this._x - 1;
            }
         }
      }
   }
   if(this.attackzl == 3 && this._currentframe < 120)
   {
      this.goal = estan;
      if(this._currentframe > 28 && this._currentframe < 120)
      {
         this.gotoAndPlay(15);
      }
      if(this._currentframe < 15 && this._currentframe < 120)
      {
         this.gotoAndPlay(15);
      }
      this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 100 - this._x));
      if(this.goal._x - 100 - this._x > 0)
      {
         this.dx = Math.cos(this.angle) * this.speed;
         this.dy = Math.sin(this.angle) * this.speed;
      }
      if(this.goal._x - 100 - this._x == 0)
      {
         if(Math.abs(this.goal._y - this._y) >= 180)
         {
            if(this.goal._y - this._y > 0)
            {
               this.dx = 0;
               this.dy = 1;
            }
            if(this.goal._y - this._y < 0)
            {
               this.dx = 0;
               this.dy = -1;
            }
         }
      }
      if(this.goal._x - 100 - this._x < 0)
      {
         this.dx = (- Math.cos(this.angle)) * this.speed;
         this.dy = (- Math.sin(this.angle)) * this.speed;
      }
      this._x = this._x + this.dx;
      this._y = this._y + this.dy;
   }
   if(this.attackzl == 0 && this._currentframe < 120)
   {
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k < 16)
         {
            changsx(this,people[k]);
         }
         k++;
      }
      this.goalxs = 0;
      this.goaljl = 120;
      i = 0;
      while(i < 30)
      {
         if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && people[i].jlife == 1 && people[i].life < people[i].xxx && i < 16)
         {
            this.goal = people[i];
            this.attackzl = 2;
            this.goalxs = this.goalxs + 1;
            this.goaljl = Math.abs(people[i]._x - this._x);
         }
         if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && i >= 16)
         {
            this.goal = people[i];
            this.attackzl = 2;
            this.goalxs = this.goalxs + 1;
            this.goaljl = Math.abs(people[i]._x - this._x);
         }
         if(i == 29 && this.goalxs == 0)
         {
            this.gotoAndStop(1);
         }
         i++;
      }
   }
}
function xds2Action()
{
   this.xue.gotoAndStop(int(this.life / this.xxx * 100));
   if(this._currentframe == 146)
   {
      ebsxlh.push(this.number);
      ecastle.unit--;
      this.removeMovieClip();
   }
   if(this.life <= 0 && this._currentframe < 120)
   {
      this.gotoAndPlay(120);
      zczk2.text = zczk.text;
      zczk.text = "- A monk of Beilmen died .";
   }
   if(this.attackzl == 1 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 30;
         while(i > 0)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && people[i].jlife == 0 && people[i].life < people[i].xxx && i >= 16)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
               break;
            }
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && i < 16)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            if(i == 1 && this.goalxs == 0)
            {
               this.goal = estan;
            }
            i--;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(Math.sqrt(Math.pow(this.goal._x - this._x,2) + Math.pow(this.goal._y - this._y,2)) < 150 && this._currentframe < 28)
      {
         if(this.goal !== ecastle && this.goal !== estan)
         {
            changlayer(this,this.goal);
         }
         this.gotoAndPlay(51);
      }
      if(this.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndPlay(120);
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.goal._currentframe < 120 && this._currentframe > 28 && Math.sqrt(Math.pow(this.goal._x - this._x,2) + Math.pow(this.goal._y - this._y,2)) < 150)
      {
         k = 16;
         while(k < 30)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 75)
         {
            if(this.goal != estan)
            {
               if(this.goal.life > this.goal.xxx && this._currentframe < 120)
               {
                  this.gotoAndStop(1);
               }
               if(this.goal.jlife == 0)
               {
                  xdsxg.attachMovie("zly",xdsxs,xdsxs);
                  xdsxg[xdsxs]._y = this.goal._y - 80;
                  xdsxg[xdsxs]._x = this.goal._x;
                  this.goal.life = this.goal.life + this.cure;
               }
               if(this.goal.jlife == 1 or this.goal.jlife == 2)
               {
                  xdsxg.attachMovie("jsgj",xdsxs,xdsxs);
                  xdsxg[xdsxs]._y = this.goal._y - Math.random() * 30;
                  xdsxg[xdsxs]._x = this.goal._x;
                  this.goal.life = this.goal.life - (7 + Math.random() * 7);
               }
               xdsxs++;
               xdsxs = xdsxs % 30;
            }
            if(this.goal == estan)
            {
               if(this.goal.life <= 0 && this._currentframe < 120)
               {
                  this.gotoAndStop(1);
               }
               xdsxg.attachMovie("jsgj",xdsxs,xdsxs);
               xdsxg[xdsxs]._y = this.goal._y + Math.random() * 20 - 10;
               xdsxg[xdsxs]._x = this.goal._x - Math.random() * 40 - 20;
               estan.life = estan.life - (7 + Math.random() * 7);
               xdsxs++;
               xdsxs = xdsxs % 30;
            }
         }
         else
         {
            this.goalxs = 0;
            this.goaljl = 650;
            i = 30;
            while(i > 0)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && people[i].jlife == 0 && people[i].life < people[i].xxx && i >= 16)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
                  break;
               }
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && i < 16)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
               }
               if(i == 1 && this.goalxs == 0)
               {
                  this.goal = estan;
               }
               i--;
            }
         }
      }
      else
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 30;
         while(i > 0)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && people[i].jlife == 0 && people[i].life < people[i].xxx && i >= 16)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
               break;
            }
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120 && i < 16)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            if(i == 1 && this.goalxs == 0)
            {
               this.goal = estan;
            }
            i--;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         if(this.goal._x < 320)
         {
            this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x + 100 - this._x));
            if(this.goal._x + 100 - this._x > 0)
            {
               this.dx = Math.cos(this.angle) * this.speed;
               this.dy = Math.sin(this.angle) * this.speed;
            }
            if(this.goal._x + 100 - this._x == 0)
            {
               if(Math.abs(this.goal._y - this._y) >= 180)
               {
                  if(this.goal._y - this._y > 0)
                  {
                     this.dx = 0;
                     this.dy = 1;
                  }
                  if(this.goal._y - this._y < 0)
                  {
                     this.dx = 0;
                     this.dy = -1;
                  }
               }
            }
            if(this.goal._x + 100 - this._x < 0)
            {
               this.dx = (- Math.cos(this.angle)) * this.speed;
               this.dy = (- Math.sin(this.angle)) * this.speed;
            }
            this._x = this._x + this.dx;
            this._y = this._y + this.dy;
         }
      }
      if(this.goal._x >= 320 && Math.sqrt(Math.pow(this.goal._x - this._x,2) + Math.pow(this.goal._y - this._y,2)) >= 150)
      {
         if(Math.abs(this.goal._y - this._y) > 4)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         if(Math.abs(this.goal._y - this._y) <= 4)
         {
            if(this.goal._x - this._x > 0)
            {
               this._x = this._x + 1;
            }
            else
            {
               this._x = this._x - 1;
            }
         }
      }
   }
   if(this.attackzl == 0)
   {
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k >= 16)
         {
            changsx(this,people[k]);
         }
         k++;
      }
   }
}
function hpAction()
{
   this.xue.gotoAndStop(int(this.life / this.xxx * 100));
   if(this._currentframe == 128)
   {
      bsxlh.push(this.number);
      estan.unit--;
      this.removeMovieClip();
   }
   if(this.attackzl == 3 && this._currentframe < 120 && this._x < 160)
   {
      this.attackzl = 0;
   }
   if(this.life <= 0 && this._currentframe < 120)
   {
      this.gotoAndPlay(120);
      zczk2.text = zczk.text;
      zczk.text = "- A cannon of Eastan fell .";
   }
   if(this.attackzl == 1 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1 or this.goal == estan)
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.goal = ecastle;
            }
            i++;
         }
         this.gotoAndPlay(2);
      }
      if(this.goal == ecastle)
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.goal = ecastle;
            }
            i++;
         }
      }
      if(Math.abs(this.goal._x - this._x) <= 240 && this._currentframe < 28)
      {
         if(Math.abs(this.goal._y - this._y) > 15)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 0.4;
            }
            else
            {
               this._y = this._y - 0.4;
            }
         }
         else
         {
            this.gotoAndPlay(51);
         }
      }
      if(this.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndPlay(120);
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.goal._currentframe < 120 && this._currentframe > 28 && Math.abs(this.goal._y - this._y) < 36 && Math.abs(this.goal._x - this._x) <= 240)
      {
         k = 1;
         while(k < 15)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 81)
         {
            this.goal.life = this.goal.life - (this.attack + Math.random() * this.attack);
            hpxg.attachMovie("bzxg",hpxs,hpxs);
            if(this.goal != ecastle)
            {
               hpxg[hpxs]._y = this.goal._y;
               hpxg[hpxs]._x = this.goal._x;
            }
            if(this.goal == ecastle)
            {
               hpxg[hpxs]._y = this.goal._y + Math.random() * 40 - 20;
               hpxg[hpxs]._x = this.goal._x + Math.random() * 40 + 20;
            }
            hpxs++;
            hpxs = hpxs % 30;
         }
         else
         {
            this.goalxs = 0;
            this.goaljl = 650;
            i = 16;
            while(i < 30)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
               }
               else if(i == 29 && this.goalxs == 0)
               {
                  this.goal = ecastle;
               }
               i++;
            }
         }
      }
      else if(this.goal._x > 380)
      {
         if(this._currentframe > 28 && this._currentframe < 85)
         {
            this.gotoAndPlay(2);
         }
         this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 210 - this._x));
         if(this.goal._x - 210 - this._x > 0)
         {
            this.dx = Math.cos(this.angle) * this.speed;
            this.dy = Math.sin(this.angle) * this.speed;
         }
         if(this.goal._x - 210 - this._x == 0)
         {
            if(Math.abs(this.goal._y - this._y) > 15)
            {
               if(this.goal._y - this._y > 0)
               {
                  this.dx = 0;
                  this.dy = 0.4;
               }
               if(this.goal._y - this._y < 0)
               {
                  this.dx = 0;
                  this.dy = -0.4;
               }
            }
         }
         if(this.goal._x - 210 - this._x < 0)
         {
            this.dx = (- Math.cos(this.angle)) * this.speed;
            this.dy = (- Math.sin(this.angle)) * this.speed;
         }
         this._x = this._x + this.dx;
         this._y = this._y + this.dy;
      }
      if(this.goal._x <= 380 && Math.abs(this.goal._x - this._x) > 240)
      {
         if(this._currentframe > 28 && this._currentframe < 85)
         {
            this.gotoAndPlay(2);
         }
         if(Math.abs(this.goal._y - this._y) > 15)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 0.4;
            }
            else
            {
               this._y = this._y - 0.4;
            }
         }
         if(Math.abs(this.goal._y - this._y) <= 15)
         {
            if(this.goal._x - this._x > 0)
            {
               this._x = this._x + 0.4;
            }
            else
            {
               this._x = this._x - 0.4;
            }
         }
      }
   }
   if(this.attackzl == 2 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 250;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.attackzl = 0;
            }
            i++;
         }
         this.gotoAndPlay(2);
      }
      if(Math.abs(this.goal._x - this._x) <= 240 && this._currentframe < 28)
      {
         if(Math.abs(this.goal._y - this._y) > 15)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 0.4;
            }
            else
            {
               this._y = this._y - 0.4;
            }
         }
         else
         {
            this.gotoAndPlay(51);
         }
      }
      if(this.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndPlay(120);
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.goal._currentframe < 120 && this._currentframe > 28 && Math.abs(this.goal._y - this._y) < 36 && Math.abs(this.goal._x - this._x) <= 240)
      {
         k = 1;
         while(k < 15)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 81)
         {
            this.goal.life = this.goal.life - (this.attack + Math.random() * this.attack);
            hpxg.attachMovie("bzxg",hpxs,hpxs);
            if(this.goal != ecastle)
            {
               hpxg[hpxs]._y = this.goal._y;
               hpxg[hpxs]._x = this.goal._x;
            }
            if(this.goal == ecastle)
            {
               hpxg[hpxs]._y = this.goal._y + Math.random() * 40 - 20;
               hpxg[hpxs]._x = this.goal._x + Math.random() * 40 + 20;
            }
            hpxs++;
            hpxs = hpxs % 30;
         }
         else
         {
            this.goalxs = 0;
            this.goaljl = 250;
            i = 16;
            while(i < 30)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
               }
               else if(i == 29 && this.goalxs == 0)
               {
                  this.attackzl = 0;
               }
               i++;
            }
         }
      }
      else if(this.goal._x > 380)
      {
         if(this._currentframe > 28 && this._currentframe < 85)
         {
            this.gotoAndPlay(2);
         }
         this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 210 - this._x));
         if(this.goal._x - 210 - this._x > 0)
         {
            this.dx = Math.cos(this.angle) * this.speed;
            this.dy = Math.sin(this.angle) * this.speed;
         }
         if(this.goal._x - 210 - this._x == 0)
         {
            if(Math.abs(this.goal._y - this._y) > 15)
            {
               if(this.goal._y - this._y > 0)
               {
                  this.dx = 0;
                  this.dy = 0.4;
               }
               if(this.goal._y - this._y < 0)
               {
                  this.dx = 0;
                  this.dy = -0.4;
               }
            }
         }
         if(this.goal._x - 210 - this._x < 0)
         {
            this.dx = (- Math.cos(this.angle)) * this.speed;
            this.dy = (- Math.sin(this.angle)) * this.speed;
         }
         this._x = this._x + this.dx;
         this._y = this._y + this.dy;
      }
      if(this.goal._x <= 380 && Math.abs(this.goal._x - this._x) > 240)
      {
         if(this._currentframe > 28 && this._currentframe < 85)
         {
            this.gotoAndPlay(2);
         }
         if(Math.abs(this.goal._y - this._y) > 15)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 0.4;
            }
            else
            {
               this._y = this._y - 0.4;
            }
         }
         if(Math.abs(this.goal._y - this._y) <= 15)
         {
            if(this.goal._x - this._x > 0)
            {
               this._x = this._x + 0.4;
            }
            else
            {
               this._x = this._x - 0.4;
            }
         }
      }
   }
   if(this.attackzl == 3 && this._currentframe < 120)
   {
      this.goal = estan;
      if(this._currentframe > 28 && this._currentframe < 120)
      {
         this.gotoAndPlay(2);
      }
      if(this._currentframe == 1)
      {
         this.gotoAndPlay(2);
      }
      this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 100 - this._x));
      if(this.goal._x - 100 - this._x > 0)
      {
         this.dx = Math.cos(this.angle) * this.speed;
         this.dy = Math.sin(this.angle) * this.speed;
      }
      if(this.goal._x - 100 - this._x == 0)
      {
         if(Math.abs(this.goal._y - this._y) >= 180)
         {
            if(this.goal._y - this._y > 0)
            {
               this.dx = 0;
               this.dy = 1;
            }
            if(this.goal._y - this._y < 0)
            {
               this.dx = 0;
               this.dy = -1;
            }
         }
      }
      if(this.goal._x - 100 - this._x < 0)
      {
         this.dx = (- Math.cos(this.angle)) * this.speed;
         this.dy = (- Math.sin(this.angle)) * this.speed;
      }
      this._x = this._x + this.dx;
      this._y = this._y + this.dy;
   }
   if(this.attackzl == 0 && this._currentframe < 120)
   {
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k < 16)
         {
            changsx(this,people[k]);
         }
         k++;
      }
      this.goalxs = 0;
      this.goaljl = 220;
      i = 16;
      while(i < 30)
      {
         if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
         {
            this.goal = people[i];
            this.attackzl = 2;
            this.goalxs = this.goalxs + 1;
            this.goaljl = Math.abs(people[i]._x - this._x);
         }
         else if(i == 29 && this.goalxs == 0)
         {
            this.gotoAndStop(1);
         }
         i++;
      }
   }
}
function hp2Action()
{
   this.xue.gotoAndStop(int(this.life / this.xxx * 100));
   if(this._currentframe == 128)
   {
      ebsxlh.push(this.number);
      ecastle.unit--;
      this.removeMovieClip();
   }
   if(this.life <= 0 && this._currentframe < 120)
   {
      this.gotoAndPlay(120);
      zczk2.text = zczk.text;
      zczk.text = "- A cannon of Beilmen fell .";
   }
   if(this.attackzl == 1 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 1;
         while(i < 15)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 14 && this.goalxs == 0)
            {
               this.goal = estan;
            }
            i++;
         }
         this.gotoAndPlay(2);
      }
      if(Math.abs(this.goal._x - this._x) <= 240 && this._currentframe < 28)
      {
         if(Math.abs(this.goal._y - this._y) > 15)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 0.4;
            }
            else
            {
               this._y = this._y - 0.4;
            }
         }
         else
         {
            this.gotoAndPlay(51);
         }
      }
      if(this.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndPlay(120);
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.goal._currentframe < 120 && this._currentframe > 28 && Math.abs(this.goal._y - this._y) < 36 && Math.abs(this.goal._x - this._x) <= 240)
      {
         k = 16;
         while(k < 30)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 81)
         {
            this.goal.life = this.goal.life - (this.attack + Math.random() * this.attack);
            hpxg.attachMovie("bzxg",hpxs,hpxs);
            if(this.goal != estan)
            {
               hpxg[hpxs]._y = this.goal._y;
               hpxg[hpxs]._x = this.goal._x;
            }
            if(this.goal == estan)
            {
               hpxg[hpxs]._y = this.goal._y + Math.random() * 40 - 20;
               hpxg[hpxs]._x = this.goal._x - Math.random() * 40 - 20;
            }
            hpxs++;
            hpxs = hpxs % 30;
         }
         else
         {
            this.goalxs = 0;
            this.goaljl = 650;
            i = 1;
            while(i < 15)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
               }
               else if(i == 14 && this.goalxs == 0)
               {
                  this.goal = estan;
               }
               i++;
            }
         }
      }
      else if(this.goal._x < 260)
      {
         if(this._currentframe > 28 && this._currentframe < 85)
         {
            this.gotoAndPlay(2);
         }
         this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x + 210 - this._x));
         if(this.goal._x + 210 - this._x > 0)
         {
            this.dx = Math.cos(this.angle) * this.speed;
            this.dy = Math.sin(this.angle) * this.speed;
         }
         if(this.goal._x + 210 - this._x == 0)
         {
            if(Math.abs(this.goal._y - this._y) > 15)
            {
               if(this.goal._y - this._y > 0)
               {
                  this.dx = 0;
                  this.dy = 0.4;
               }
               if(this.goal._y - this._y < 0)
               {
                  this.dx = 0;
                  this.dy = -0.4;
               }
            }
         }
         if(this.goal._x + 210 - this._x < 0)
         {
            this.dx = (- Math.cos(this.angle)) * this.speed;
            this.dy = (- Math.sin(this.angle)) * this.speed;
         }
         this._x = this._x + this.dx;
         this._y = this._y + this.dy;
      }
      if(this.goal._x >= 260 && Math.abs(this.goal._x - this._x) > 240)
      {
         if(this._currentframe > 28 && this._currentframe < 85)
         {
            this.gotoAndPlay(2);
         }
         if(Math.abs(this.goal._y - this._y) > 15)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 0.4;
            }
            else
            {
               this._y = this._y - 0.4;
            }
         }
         if(Math.abs(this.goal._y - this._y) <= 15)
         {
            if(this.goal._x - this._x > 0)
            {
               this._x = this._x + 0.4;
            }
            else
            {
               this._x = this._x - 0.4;
            }
         }
      }
   }
   if(this.attackzl == 0)
   {
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k >= 16)
         {
            changsx(this,people[k]);
         }
         k++;
      }
   }
}
function fsAction()
{
   this.xue.gotoAndStop(int(this.life / this.xxx * 100));
   if(this._currentframe == 146)
   {
      bsxlh.push(this.number);
      estan.unit--;
      this.removeMovieClip();
   }
   if(this.attackzl == 3 && this._currentframe < 120 && this._x < 160)
   {
      this.gotoAndStop(1);
      this.goal = ecastle;
      this.attackzl = 0;
   }
   if(this.life <= 0 && this._currentframe < 120)
   {
      this.gotoAndPlay(120);
      zczk2.text = zczk.text;
      zczk.text = "- A magician of Eastan died .";
   }
   if(this.attackzl == 1 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 0;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) >= this.goaljl && people[i]._currentframe < 120 && people[i].attackzl == 1)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               j = 16;
               while(j < 30)
               {
                  if(Math.abs(people[j]._x - this._x) >= this.goaljl && people[j]._currentframe < 120)
                  {
                     this.goal = people[j];
                     this.goalxs = this.goalxs + 1;
                     this.goaljl = Math.abs(people[j]._x - this._x);
                  }
                  else if(j == 29 && this.goalxs == 0)
                  {
                     this.goal = ecastle;
                  }
                  j++;
               }
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(Math.abs(this.goal._x - this._x) < 250 && this._currentframe < 28)
      {
         if(this.goal !== ecastle && this.goal !== estan)
         {
            changlayer(this,this.goal);
         }
         this.gotoAndPlay(51);
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.goal._currentframe < 120 && this._currentframe > 28 && Math.abs(this.goal._x - this._x) < 250)
      {
         k = 1;
         while(k < 15)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 73)
         {
            xdsxg.attachMovie("fsxg",xdsxs,xdsxs);
            xdsxg[xdsxs]._y = this.goal._y - 30;
            xdsxg[xdsxs]._x = this.goal._x;
            xdsxs++;
            xdsxs = xdsxs % 30;
         }
         if(this._currentframe == 89)
         {
            this.goal.life = this.goal.life - (this.attack + Math.random() * this.attack);
         }
         if(this.goal == ecastle)
         {
            this.goalxs = 0;
            this.goaljl = 0;
            i = 16;
            while(i < 30)
            {
               if(Math.abs(people[i]._x - this._x) >= this.goaljl && people[i]._currentframe < 120 && people[i].attackzl == 1)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
               }
               else if(i == 29 && this.goalxs == 0)
               {
                  j = 16;
                  while(j < 30)
                  {
                     if(Math.abs(people[j]._x - this._x) >= this.goaljl && people[j]._currentframe < 120)
                     {
                        this.goal = people[j];
                        this.goalxs = this.goalxs + 1;
                        this.goaljl = Math.abs(people[j]._x - this._x);
                     }
                     else if(j == 29 && this.goalxs == 0)
                     {
                        this.goal = ecastle;
                     }
                     j++;
                  }
               }
               i++;
            }
         }
      }
      else
      {
         this.goalxs = 0;
         this.goaljl = 0;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) >= this.goaljl && people[i]._currentframe < 120 && people[i].attackzl == 1)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               j = 16;
               while(j < 30)
               {
                  if(Math.abs(people[j]._x - this._x) >= this.goaljl && people[j]._currentframe < 120)
                  {
                     this.goal = people[j];
                     this.goalxs = this.goalxs + 1;
                     this.goaljl = Math.abs(people[j]._x - this._x);
                  }
                  else if(j == 29 && this.goalxs == 0)
                  {
                     this.goal = ecastle;
                  }
                  j++;
               }
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         if(this.goal._x > 320)
         {
            this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 150 - this._x));
            if(this.goal._x - 150 - this._x > 0)
            {
               this.dx = Math.cos(this.angle) * this.speed;
               this.dy = Math.sin(this.angle) * this.speed;
            }
            if(this.goal._x - 150 - this._x == 0)
            {
               if(Math.abs(this.goal._y - this._y) > 8)
               {
                  if(this.goal._y - this._y > 0)
                  {
                     this.dx = 0;
                     this.dy = 1;
                  }
                  if(this.goal._y - this._y < 0)
                  {
                     this.dx = 0;
                     this.dy = -1;
                  }
               }
            }
            if(this.goal._x - 150 - this._x < 0)
            {
               this.dx = (- Math.cos(this.angle)) * this.speed;
               this.dy = (- Math.sin(this.angle)) * this.speed;
            }
            this._x = this._x + this.dx;
            this._y = this._y + this.dy;
         }
      }
   }
   if(this.attackzl == 2 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 0;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) >= this.goaljl && people[i]._currentframe < 120 && people[i].attackzl == 1)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               j = 16;
               while(j < 30)
               {
                  if(Math.abs(people[j]._x - this._x) >= this.goaljl && people[j]._currentframe < 120)
                  {
                     this.goal = people[j];
                     this.goalxs = this.goalxs + 1;
                     this.goaljl = Math.abs(people[j]._x - this._x);
                  }
                  else if(j == 29 && this.goalxs == 0)
                  {
                     this.attackzl = 0;
                  }
                  j++;
               }
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(Math.abs(this.goal._x - this._x) < 250 && this._currentframe < 28)
      {
         if(this.goal !== ecastle && this.goal !== estan)
         {
            changlayer(this,this.goal);
         }
         this.gotoAndPlay(51);
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.goal._currentframe < 120 && this._currentframe > 28 && Math.abs(this.goal._x - this._x) < 250)
      {
         k = 1;
         while(k < 15)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 73)
         {
            xdsxg.attachMovie("fsxg",xdsxs,xdsxs);
            xdsxg[xdsxs]._y = this.goal._y - 30;
            xdsxg[xdsxs]._x = this.goal._x;
            xdsxs++;
            xdsxs = xdsxs % 30;
         }
         if(this._currentframe == 89)
         {
            this.goal.life = this.goal.life - (this.attack + Math.random() * this.attack);
         }
         if(this.goal == ecastle)
         {
            this.goalxs = 0;
            this.goaljl = 0;
            i = 16;
            while(i < 30)
            {
               if(Math.abs(people[i]._x - this._x) >= this.goaljl && people[i]._currentframe < 120 && people[i].attackzl == 1)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
               }
               else if(i == 29 && this.goalxs == 0)
               {
                  j = 16;
                  while(j < 30)
                  {
                     if(Math.abs(people[j]._x - this._x) >= this.goaljl && people[j]._currentframe < 120)
                     {
                        this.goal = people[j];
                        this.goalxs = this.goalxs + 1;
                        this.goaljl = Math.abs(people[j]._x - this._x);
                     }
                     else if(j == 29 && this.goalxs == 0)
                     {
                        this.attackzl = 0;
                     }
                     j++;
                  }
               }
               i++;
            }
         }
      }
      else
      {
         this.goalxs = 0;
         this.goaljl = 0;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) >= this.goaljl && people[i]._currentframe < 120 && people[i].attackzl == 1)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               j = 16;
               while(j < 30)
               {
                  if(Math.abs(people[j]._x - this._x) >= this.goaljl && people[j]._currentframe < 120)
                  {
                     this.goal = people[j];
                     this.goalxs = this.goalxs + 1;
                     this.goaljl = Math.abs(people[j]._x - this._x);
                  }
                  else if(j == 29 && this.goalxs == 0)
                  {
                     this.attackzl = 0;
                  }
                  j++;
               }
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         if(this.goal._x > 320)
         {
            this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 150 - this._x));
            if(this.goal._x - 150 - this._x > 0)
            {
               this.dx = Math.cos(this.angle) * this.speed;
               this.dy = Math.sin(this.angle) * this.speed;
            }
            if(this.goal._x - 150 - this._x == 0)
            {
               if(Math.abs(this.goal._y - this._y) > 8)
               {
                  if(this.goal._y - this._y > 0)
                  {
                     this.dx = 0;
                     this.dy = 1;
                  }
                  if(this.goal._y - this._y < 0)
                  {
                     this.dx = 0;
                     this.dy = -1;
                  }
               }
            }
            if(this.goal._x - 150 - this._x < 0)
            {
               this.dx = (- Math.cos(this.angle)) * this.speed;
               this.dy = (- Math.sin(this.angle)) * this.speed;
            }
            this._x = this._x + this.dx;
            this._y = this._y + this.dy;
         }
      }
   }
   if(this.attackzl == 3 && this._currentframe < 120)
   {
      this.goal = estan;
      if(this._currentframe > 28 && this._currentframe < 120)
      {
         this.gotoAndPlay(15);
      }
      if(this._currentframe < 15 && this._currentframe < 120)
      {
         this.gotoAndPlay(15);
      }
      this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 100 - this._x));
      if(this.goal._x - 100 - this._x > 0)
      {
         this.dx = Math.cos(this.angle) * this.speed;
         this.dy = Math.sin(this.angle) * this.speed;
      }
      if(this.goal._x - 100 - this._x == 0)
      {
         if(Math.abs(this.goal._y - this._y) >= 180)
         {
            if(this.goal._y - this._y > 0)
            {
               this.dx = 0;
               this.dy = 1;
            }
            if(this.goal._y - this._y < 0)
            {
               this.dx = 0;
               this.dy = -1;
            }
         }
      }
      if(this.goal._x - 100 - this._x < 0)
      {
         this.dx = (- Math.cos(this.angle)) * this.speed;
         this.dy = (- Math.sin(this.angle)) * this.speed;
      }
      this._x = this._x + this.dx;
      this._y = this._y + this.dy;
   }
   if(this.attackzl == 0 && this._currentframe < 120)
   {
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k < 16)
         {
            changsx(this,people[k]);
         }
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 0;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) >= this.goaljl && people[i]._currentframe < 120 && people[i]._x < 340)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.attackzl = 2;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.gotoAndStop(1);
            }
            i++;
         }
      }
   }
}
function fs2Action()
{
   this.xue.gotoAndStop(int(this.life / this.xxx * 100));
   if(this._currentframe == 146)
   {
      ebsxlh.push(this.number);
      ecastle.unit--;
      this.removeMovieClip();
   }
   if(this.life <= 0 && this._currentframe < 120)
   {
      this.gotoAndPlay(120);
      zczk2.text = zczk.text;
      zczk.text = "- A magician of Beilmen died .";
   }
   if(this.attackzl == 1 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 0;
         i = 1;
         while(i < 15)
         {
            if(Math.abs(people[i]._x - this._x) > this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 14 && this.goalxs == 0)
            {
               this.goal = estan;
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(Math.abs(this.goal._x - this._x) < 250 && this._currentframe < 28)
      {
         if(this.goal != ecastle && this.goal != estan)
         {
            changlayer(this,this.goal);
         }
         if(Math.abs(this.goal._y - this._y) >= 50)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         else
         {
            this.gotoAndPlay(51);
         }
      }
      if(this.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndPlay(120);
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.goal._currentframe < 120 && this._currentframe > 28 && Math.abs(this.goal._y - this._y) < 50 && Math.abs(this.goal._x - this._x) < 250)
      {
         k = 16;
         while(k < 30)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 73)
         {
            xdsxg.attachMovie("fsxg2",xdsxs,xdsxs);
            xdsxg[xdsxs]._y = this.goal._y - 30;
            xdsxg[xdsxs]._x = this.goal._x;
            xdsxs++;
            xdsxs = xdsxs % 30;
         }
         if(this._currentframe == 89)
         {
            this.goal.life = this.goal.life - (this.attack + Math.random() * this.attack);
         }
         if(this.goal == estan)
         {
            this.goalxs = 0;
            this.goaljl = 0;
            i = 1;
            while(i < 15)
            {
               if(Math.abs(people[i]._x - this._x) > this.goaljl && people[i]._currentframe < 120)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
               }
               else if(i == 14 && this.goalxs == 0)
               {
                  this.goal = estan;
               }
               i++;
            }
         }
      }
      else
      {
         this.goalxs = 0;
         this.goaljl = 0;
         i = 1;
         while(i < 15)
         {
            if(Math.abs(people[i]._x - this._x) > this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 14 && this.goalxs == 0)
            {
               this.goal = estan;
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         if(this.goal._x < 320)
         {
            this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x + 150 - this._x));
            if(this.goal._x + 150 - this._x > 0)
            {
               this.dx = Math.cos(this.angle) * this.speed;
               this.dy = Math.sin(this.angle) * this.speed;
            }
            if(this.goal._x + 150 - this._x == 0)
            {
               if(Math.abs(this.goal._y - this._y) >= 50)
               {
                  if(this.goal._y - this._y > 0)
                  {
                     this.dx = 0;
                     this.dy = 1;
                  }
                  if(this.goal._y - this._y < 0)
                  {
                     this.dx = 0;
                     this.dy = -1;
                  }
               }
            }
            if(this.goal._x + 150 - this._x < 0)
            {
               this.dx = (- Math.cos(this.angle)) * this.speed;
               this.dy = (- Math.sin(this.angle)) * this.speed;
            }
            this._x = this._x + this.dx;
            this._y = this._y + this.dy;
         }
      }
   }
   if(this.attackzl == 0 && this._currentframe < 120)
   {
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k >= 16)
         {
            changsx(this,people[k]);
         }
         k++;
      }
   }
}
function hqsAction()
{
   this.xue.gotoAndStop(int(this.life / this.xxx * 100));
   if(this._currentframe == 146)
   {
      bsxlh.push(this.number);
      estan.unit--;
      if(this.heroz == 1)
      {
         lordan.gotoAndStop(1);
      }
      this.removeMovieClip();
   }
   if(this.attackzl == 3 && this._currentframe < 120 && this._x < 160)
   {
      this.attackzl = 0;
   }
   if(this.life < this.xxx && this.heroz == 1)
   {
      this.life = this.life + 0.3;
   }
   if(this.life <= 0 && this._currentframe < 120)
   {
      this.gotoAndPlay(120);
      if(this.heroz == 1)
      {
         zczk2.text = zczk.text;
         zczk.text = "- The hero of Eastan died .";
      }
      else
      {
         zczk2.text = zczk.text;
         zczk.text = "- A rifleman of Eastan died .";
      }
   }
   if(this.attackzl == 1 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.goal = ecastle;
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(Math.abs(this.goal._x - this._x) < 160 && this._currentframe < 28)
      {
         if(this.goal !== ecastle && this.goal !== estan)
         {
            changlayer(this,this.goal);
         }
         if(Math.abs(this.goal._y - this._y) > 8)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         else
         {
            if(this._x < this.goal._x)
            {
               this.gotoAndPlay(51);
            }
            if(this._x == this.goal._x)
            {
               if(this._y > this.goal._y)
               {
                  this.gotoAndPlay(51);
               }
               if(this._y < this.goal._y)
               {
                  this.gotoAndPlay(88);
               }
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(88);
            }
         }
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.goal._currentframe < 120 && this._currentframe > 28 && Math.abs(this.goal._y - this._y) < 36 && Math.abs(this.goal._x - this._x) < 160)
      {
         k = 1;
         while(k < 15)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 62 or this._currentframe == 91)
         {
            this.goal.life = this.goal.life - (this.attack + Math.random() * this.attack);
         }
         else
         {
            this.goalxs = 0;
            this.goaljl = 650;
            i = 16;
            while(i < 30)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
               }
               else if(i == 29 && this.goalxs == 0)
               {
                  this.goal = ecastle;
               }
               i++;
            }
         }
      }
      else
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.goal = ecastle;
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         if(this.goal._x > 320)
         {
            this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 150 - this._x));
            if(this.goal._x - 150 - this._x > 0)
            {
               this.dx = Math.cos(this.angle) * this.speed;
               this.dy = Math.sin(this.angle) * this.speed;
            }
            if(this.goal._x - 150 - this._x == 0)
            {
               if(Math.abs(this.goal._y - this._y) > 8)
               {
                  if(this.goal._y - this._y > 0)
                  {
                     this.dx = 0;
                     this.dy = 1;
                  }
                  if(this.goal._y - this._y < 0)
                  {
                     this.dx = 0;
                     this.dy = -1;
                  }
               }
            }
            if(this.goal._x - 150 - this._x < 0)
            {
               this.dx = (- Math.cos(this.angle)) * this.speed;
               this.dy = (- Math.sin(this.angle)) * this.speed;
            }
            this._x = this._x + this.dx;
            this._y = this._y + this.dy;
         }
      }
      if(this.goal._x <= 320 && Math.abs(this.goal._x - this._x) >= 160)
      {
         if(Math.abs(this.goal._y - this._y) > 8)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         if(Math.abs(this.goal._y - this._y) <= 8)
         {
            if(this.goal._x - this._x > 0)
            {
               this._x = this._x + 1;
            }
            else
            {
               this._x = this._x - 1;
            }
         }
      }
   }
   if(this.attackzl == 2 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 220;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.attackzl = 0;
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(Math.abs(this.goal._x - this._x) < 160 && this._currentframe < 28)
      {
         if(this.goal !== ecastle && this.goal !== estan)
         {
            changlayer(this,this.goal);
         }
         if(Math.abs(this.goal._y - this._y) > 8)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         else
         {
            if(this._x < this.goal._x)
            {
               this.gotoAndPlay(51);
            }
            if(this._x == this.goal._x)
            {
               if(this._y > this.goal._y)
               {
                  this.gotoAndPlay(51);
               }
               if(this._y < this.goal._y)
               {
                  this.gotoAndPlay(88);
               }
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(88);
            }
         }
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.goal._currentframe < 120 && this._currentframe > 28 && Math.abs(this.goal._y - this._y) < 36 && Math.abs(this.goal._x - this._x) < 160)
      {
         k = 1;
         while(k < 15)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 62 or this._currentframe == 91)
         {
            this.goal.life = this.goal.life - (this.attack + Math.random() * this.attack);
         }
         else
         {
            this.goalxs = 0;
            this.goaljl = 220;
            i = 16;
            while(i < 30)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
               {
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
                  this.goaljl = Math.abs(people[i]._x - this._x);
               }
               else if(i == 29 && this.goalxs == 0)
               {
                  this.attackzl = 0;
               }
               i++;
            }
         }
      }
      else
      {
         this.goalxs = 0;
         this.goaljl = 220;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.attackzl = 0;
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         if(this.goal._x > 320)
         {
            this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 150 - this._x));
            if(this.goal._x - 150 - this._x > 0)
            {
               this.dx = Math.cos(this.angle) * this.speed;
               this.dy = Math.sin(this.angle) * this.speed;
            }
            if(this.goal._x - 150 - this._x == 0)
            {
               if(Math.abs(this.goal._y - this._y) > 8)
               {
                  if(this.goal._y - this._y > 0)
                  {
                     this.dx = 0;
                     this.dy = 1;
                  }
                  if(this.goal._y - this._y < 0)
                  {
                     this.dx = 0;
                     this.dy = -1;
                  }
               }
            }
            if(this.goal._x - 150 - this._x < 0)
            {
               this.dx = (- Math.cos(this.angle)) * this.speed;
               this.dy = (- Math.sin(this.angle)) * this.speed;
            }
            this._x = this._x + this.dx;
            this._y = this._y + this.dy;
         }
      }
      if(this.goal._x <= 320 && Math.abs(this.goal._x - this._x) >= 160)
      {
         if(Math.abs(this.goal._y - this._y) > 8)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         if(Math.abs(this.goal._y - this._y) <= 8)
         {
            if(this.goal._x - this._x > 0)
            {
               this._x = this._x + 1;
            }
            else
            {
               this._x = this._x - 1;
            }
         }
      }
   }
   if(this.attackzl == 3 && this._currentframe < 120)
   {
      this.goal = estan;
      if(this._currentframe > 28 && this._currentframe < 120)
      {
         this.gotoAndPlay(15);
      }
      if(this._currentframe < 15 && this._currentframe < 120)
      {
         this.gotoAndPlay(15);
      }
      this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 100 - this._x));
      if(this.goal._x - 100 - this._x > 0)
      {
         this.dx = Math.cos(this.angle) * this.speed;
         this.dy = Math.sin(this.angle) * this.speed;
      }
      if(this.goal._x - 100 - this._x == 0)
      {
         if(Math.abs(this.goal._y - this._y) >= 180)
         {
            if(this.goal._y - this._y > 0)
            {
               this.dx = 0;
               this.dy = 1;
            }
            if(this.goal._y - this._y < 0)
            {
               this.dx = 0;
               this.dy = -1;
            }
         }
      }
      if(this.goal._x - 100 - this._x < 0)
      {
         this.dx = (- Math.cos(this.angle)) * this.speed;
         this.dy = (- Math.sin(this.angle)) * this.speed;
      }
      this._x = this._x + this.dx;
      this._y = this._y + this.dy;
   }
   if(this.attackzl == 0 && this._currentframe < 120)
   {
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k < 16)
         {
            changsx(this,people[k]);
         }
         k++;
      }
      this.goalxs = 0;
      this.goaljl = 220;
      i = 16;
      while(i < 30)
      {
         if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
         {
            this.goal = people[i];
            this.attackzl = 2;
            this.goalxs = this.goalxs + 1;
            this.goaljl = Math.abs(people[i]._x - this._x);
         }
         else if(i == 29 && this.goalxs == 0)
         {
            this.gotoAndStop(1);
         }
         i++;
      }
   }
}
function hqs2Action()
{
   this.xue.gotoAndStop(int(this.life / this.xxx * 100));
   if(this._currentframe == 146)
   {
      ebsxlh.push(this.number);
      ecastle.unit--;
      this.removeMovieClip();
   }
   if(this.life <= 0 && this._currentframe < 120)
   {
      this.gotoAndPlay(120);
      zczk2.text = zczk.text;
      zczk.text = "- A rifleman of Beilmen died .";
   }
   if(this.attackzl == 1 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 1;
         while(i < 15)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 14 && this.goalxs == 0)
            {
               this.goal = estan;
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(Math.abs(this.goal._x - this._x) < 160 && this._currentframe < 28)
      {
         if(this.goal != ecastle && this.goal != estan)
         {
            changlayer(this,this.goal);
         }
         if(Math.abs(this.goal._y - this._y) > 8)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         else
         {
            if(this._x < this.goal._x)
            {
               this.gotoAndPlay(51);
            }
            if(this._x == this.goal._x)
            {
               if(this._y > this.goal._y)
               {
                  this.gotoAndPlay(51);
               }
               if(this._y < this.goal._y)
               {
                  this.gotoAndPlay(88);
               }
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(88);
            }
         }
      }
      if(this.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndPlay(120);
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.goal._currentframe < 120 && this._currentframe > 28 && Math.abs(this.goal._y - this._y) < 36 && Math.abs(this.goal._x - this._x) < 160)
      {
         k = 16;
         while(k < 30)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 62 or this._currentframe == 91)
         {
            this.goal.life = this.goal.life - (this.attack + Math.random() * this.attack);
         }
         else
         {
            this.goalxs = 0;
            this.goaljl = 650;
            i = 1;
            while(i < 15)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
               {
                  this.goaljl = Math.abs(people[i]._x - this._x);
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
               }
               else if(i == 14 && this.goalxs == 0)
               {
                  this.goal = estan;
               }
               i++;
            }
         }
      }
      else
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 1;
         while(i < 15)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 14 && this.goalxs == 0)
            {
               this.goal = estan;
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         if(this.goal._x < 320)
         {
            this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x + 150 - this._x));
            if(this.goal._x + 150 - this._x > 0)
            {
               this.dx = Math.cos(this.angle) * this.speed;
               this.dy = Math.sin(this.angle) * this.speed;
            }
            if(this.goal._x + 150 - this._x == 0)
            {
               if(Math.abs(this.goal._y - this._y) > 8)
               {
                  if(this.goal._y - this._y > 0)
                  {
                     this.dx = 0;
                     this.dy = 1;
                  }
                  if(this.goal._y - this._y < 0)
                  {
                     this.dx = 0;
                     this.dy = -1;
                  }
               }
            }
            if(this.goal._x + 150 - this._x < 0)
            {
               this.dx = (- Math.cos(this.angle)) * this.speed;
               this.dy = (- Math.sin(this.angle)) * this.speed;
            }
            this._x = this._x + this.dx;
            this._y = this._y + this.dy;
         }
         if(this.goal._x >= 320 && Math.abs(this.goal._x - this._x) >= 160)
         {
            if(Math.abs(this.goal._y - this._y) > 8)
            {
               if(this.goal._y - this._y > 0)
               {
                  this._y = this._y + 1;
               }
               else
               {
                  this._y = this._y - 1;
               }
            }
            if(Math.abs(this.goal._y - this._y) <= 8)
            {
               if(this.goal._x - this._x > 0)
               {
                  this._x = this._x + 1;
               }
               else
               {
                  this._x = this._x - 1;
               }
            }
         }
      }
   }
   if(this.attackzl == 0 && this._currentframe < 120)
   {
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k >= 16)
         {
            changsx(this,people[k]);
         }
         k++;
      }
   }
}
function jydAction()
{
   this.xue.gotoAndStop(int(this.life / this.xxx * 100));
   if(this._currentframe == 146)
   {
      bsxlh.push(this.number);
      estan.unit--;
      jydan.gotoAndPlay(1);
      this.removeMovieClip();
   }
   if(this.attackzl == 3 && this._currentframe < 120 && this._x < 160)
   {
      this.attackzl = 0;
   }
   if(this.life <= 0 && this._currentframe < 120)
   {
      this.gotoAndPlay(120);
      k = 0;
      while(k < 16)
      {
         people[k].dbj.gotoAndStop(1);
         k++;
      }
      zczk2.text = zczk.text;
      zczk.text = "- A military band of Eastan fell .";
   }
   if(this.attackzl == 1 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k < 16)
         {
            if(people[k].jx == 0)
            {
               people[k].life = people[k].life + 30;
               people[k].xxx = people[k].xxx + 30;
               people[k].dbj.gotoAndStop(2);
               people[k].jx = 2;
            }
            if(people[k].jx == 2 && people[k].dbj._currentframe == 1)
            {
               people[k].dbj.gotoAndStop(2);
            }
         }
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.goal = ecastle;
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(Math.abs(this.goal._x - this._x) < 190 && this._currentframe < 28)
      {
         if(this.goal !== ecastle && this.goal !== estan)
         {
            changlayer(this,this.goal);
         }
         if(Math.abs(this.goal._y - this._y) >= 50)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         else
         {
            this.gotoAndPlay(51);
         }
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(!(this.goal._currentframe < 120 && this._currentframe > 28 && Math.abs(this.goal._y - this._y) < 50 && Math.abs(this.goal._x - this._x) < 190))
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.goal = ecastle;
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         if(this.goal._x > 340)
         {
            this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 170 - this._x));
            if(this.goal._x - 170 - this._x > 0)
            {
               this.dx = Math.cos(this.angle) * this.speed;
               this.dy = Math.sin(this.angle) * this.speed;
            }
            if(this.goal._x - 170 - this._x == 0)
            {
               if(Math.abs(this.goal._y - this._y) > 8)
               {
                  if(this.goal._y - this._y > 0)
                  {
                     this.dx = 0;
                     this.dy = 1;
                  }
                  if(this.goal._y - this._y < 0)
                  {
                     this.dx = 0;
                     this.dy = -1;
                  }
               }
            }
            if(this.goal._x - 170 - this._x < 0)
            {
               this.dx = (- Math.cos(this.angle)) * this.speed;
               this.dy = (- Math.sin(this.angle)) * this.speed;
            }
            this._x = this._x + this.dx;
            this._y = this._y + this.dy;
         }
      }
   }
   if(this.attackzl == 2 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k < 16)
         {
            if(people[k].jx == 0)
            {
               people[k].life = people[k].life + 30;
               people[k].xxx = people[k].xxx + 30;
               people[k].dbj.gotoAndStop(2);
               people[k].jx = 2;
            }
            if(people[k].jx == 2 && people[k].dbj._currentframe == 1)
            {
               people[k].dbj.gotoAndStop(2);
            }
         }
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 220;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.attackzl = 0;
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(Math.abs(this.goal._x - this._x) < 190 && this._currentframe < 28)
      {
         if(this.goal !== ecastle && this.goal !== estan)
         {
            changlayer(this,this.goal);
         }
         if(Math.abs(this.goal._y - this._y) >= 50)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         else
         {
            this.gotoAndPlay(51);
         }
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(!(this.goal._currentframe < 120 && this._currentframe > 28 && Math.abs(this.goal._y - this._y) < 50 && Math.abs(this.goal._x - this._x) < 190))
      {
         this.goalxs = 0;
         this.goaljl = 220;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.atackzl = 0;
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         if(this.goal._x > 340)
         {
            this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 170 - this._x));
            if(this.goal._x - 170 - this._x > 0)
            {
               this.dx = Math.cos(this.angle) * this.speed;
               this.dy = Math.sin(this.angle) * this.speed;
            }
            if(this.goal._x - 170 - this._x == 0)
            {
               if(Math.abs(this.goal._y - this._y) > 8)
               {
                  if(this.goal._y - this._y > 0)
                  {
                     this.dx = 0;
                     this.dy = 1;
                  }
                  if(this.goal._y - this._y < 0)
                  {
                     this.dx = 0;
                     this.dy = -1;
                  }
               }
            }
            if(this.goal._x - 170 - this._x < 0)
            {
               this.dx = (- Math.cos(this.angle)) * this.speed;
               this.dy = (- Math.sin(this.angle)) * this.speed;
            }
            this._x = this._x + this.dx;
            this._y = this._y + this.dy;
         }
      }
   }
   if(this.attackzl == 3 && this._currentframe < 120)
   {
      this.goal = estan;
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k < 16)
         {
            if(people[k].jx == 0)
            {
               people[k].life = people[k].life + 30;
               people[k].xxx = people[k].xxx + 30;
               people[k].dbj.gotoAndStop(2);
               people[k].jx = 2;
            }
            if(people[k].jx == 2 && people[k].dbj._currentframe == 1)
            {
               people[k].dbj.gotoAndStop(2);
            }
         }
         k++;
      }
      if(this._currentframe > 28 && this._currentframe < 120)
      {
         this.gotoAndPlay(15);
      }
      if(this._currentframe < 15 && this._currentframe < 120)
      {
         this.gotoAndPlay(15);
      }
      this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 120 - this._x));
      if(this.goal._x - 120 - this._x > 0)
      {
         this.dx = Math.cos(this.angle) * this.speed;
         this.dy = Math.sin(this.angle) * this.speed;
      }
      if(this.goal._x - 120 - this._x == 0)
      {
         if(Math.abs(this.goal._y - this._y) >= 180)
         {
            if(this.goal._y - this._y > 0)
            {
               this.dx = 0;
               this.dy = 1;
            }
            if(this.goal._y - this._y < 0)
            {
               this.dx = 0;
               this.dy = -1;
            }
         }
      }
      if(this.goal._x - 120 - this._x < 0)
      {
         this.dx = (- Math.cos(this.angle)) * this.speed;
         this.dy = (- Math.sin(this.angle)) * this.speed;
      }
      this._x = this._x + this.dx;
      this._y = this._y + this.dy;
   }
   if(this.attackzl == 0 && this._currentframe < 120)
   {
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k < 16)
         {
            if(people[k].jx == 0)
            {
               people[k].life = people[k].life + 30;
               people[k].xxx = people[k].xxx + 30;
               people[k].dbj.gotoAndStop(2);
               people[k].jx = 2;
            }
            if(people[k].jx == 2 && people[k].dbj._currentframe == 1)
            {
               people[k].dbj.gotoAndStop(2);
            }
         }
         k++;
      }
      this.goalxs = 0;
      this.goaljl = 220;
      i = 16;
      while(i < 30)
      {
         if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
         {
            this.goal = people[i];
            this.attackzl = 2;
            this.goalxs = this.goalxs + 1;
            this.goaljl = Math.abs(people[i]._x - this._x);
         }
         else if(i == 29 && this.goalxs == 0)
         {
            this.gotoAndStop(1);
         }
         i++;
      }
   }
}
function jyd2Action()
{
   this.xue.gotoAndStop(int(this.life / this.xxx * 100));
   if(this._currentframe == 146)
   {
      ebsxlh.push(this.number);
      ecastle.unit--;
      ecastle.h4 = 1;
      hero3xl = 1;
      this.removeMovieClip();
   }
   if(this.life <= 0 && this._currentframe < 120)
   {
      this.gotoAndPlay(120);
      k = 15;
      while(k < 30)
      {
         people[k].dbj.gotoAndStop(1);
         k++;
      }
      zczk2.text = zczk.text;
      zczk.text = "- A military band of Beilmen fell .";
   }
   if(this.attackzl == 1 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         if(k > 14)
         {
            if(people[k].jx == 0)
            {
               people[k].life = people[k].life + 30;
               people[k].xxx = people[k].xxx + 30;
               people[k].dbj.gotoAndStop(2);
               people[k].jx = 2;
            }
            if(people[k].jx == 2 && people[k].dbj._currentframe == 1)
            {
               people[k].dbj.gotoAndStop(2);
            }
         }
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 1;
         while(i < 15)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 14 && this.goalxs == 0)
            {
               this.goal = estan;
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(Math.abs(this.goal._x - this._x) < 190 && this._currentframe < 28)
      {
         if(this.goal != ecastle && this.goal != estan)
         {
            changlayer(this,this.goal);
         }
         if(Math.abs(this.goal._y - this._y) >= 50)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         else
         {
            this.gotoAndPlay(88);
         }
      }
      if(this.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndPlay(120);
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.goal._currentframe < 120 && this._currentframe > 28 && Math.abs(this.goal._y - this._y) < 50 && Math.abs(this.goal._x - this._x) < 190)
      {
         k = 16;
         while(k < 30)
         {
            if(people[k].jx == 0)
            {
               people[k].life = people[k].life + 30;
               people[k].xxx = people[k].xxx + 30;
               people[k].dbj.gotoAndStop(2);
               people[k].jx = 2;
            }
            if(people[k].jx == 2 && people[k].dbj._currentframe == 1)
            {
               people[k].dbj.gotoAndStop(2);
            }
            k++;
         }
         this.goalxs = 0;
         this.goaljl = 650;
      }
      else
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 1;
         while(i < 15)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
               this.goaljl = Math.abs(people[i]._x - this._x);
            }
            else if(i == 14 && this.goalxs == 0)
            {
               this.goal = estan;
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         if(this.goal._x < 320)
         {
            this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x + 170 - this._x));
            if(this.goal._x + 170 - this._x > 0)
            {
               this.dx = Math.cos(this.angle) * this.speed;
               this.dy = Math.sin(this.angle) * this.speed;
            }
            if(this.goal._x + 170 - this._x == 0)
            {
               if(Math.abs(this.goal._y - this._y) >= 50)
               {
                  if(this.goal._y - this._y > 0)
                  {
                     this.dx = 0;
                     this.dy = 1;
                  }
                  if(this.goal._y - this._y < 0)
                  {
                     this.dx = 0;
                     this.dy = -1;
                  }
               }
            }
            if(this.goal._x + 170 - this._x < 0)
            {
               this.dx = (- Math.cos(this.angle)) * this.speed;
               this.dy = (- Math.sin(this.angle)) * this.speed;
            }
            this._x = this._x + this.dx;
            this._y = this._y + this.dy;
         }
      }
   }
   if(this.attackzl == 0 && this._currentframe < 120)
   {
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k > 14)
         {
            if(people[k].jx == 0)
            {
               people[k].life = people[k].life + 30;
               people[k].xxx = people[k].xxx + 30;
               people[k].dbj.gotoAndStop(2);
               people[k].jx = 2;
            }
            if(people[k].jx == 2 && people[k].dbj._currentframe == 1)
            {
               people[k].dbj.gotoAndStop(2);
            }
         }
         k++;
      }
   }
}
function jsAction()
{
   if(this._currentframe == 146)
   {
      bsxlh.push(this.number);
      if(this.heroz == 1)
      {
         heroan.gotoAndStop(1);
      }
      if(this.jxb == 1)
      {
         jxban.gotoAndStop(1);
      }
      estan.unit--;
      this.removeMovieClip();
   }
   if(this.attackzl == 3 && this._currentframe < 120 && this._x < 210)
   {
      this.attackzl = 0;
   }
   if(this.life < this.xxx && this.heroz == 1)
   {
      this.life = this.life + 0.3;
   }
   if(this.life <= 0 && this._currentframe < 120)
   {
      this.gotoAndPlay(120);
      if(this.heroz == 1)
      {
         zczk2.text = zczk.text;
         zczk.text = "- The hero of Eastan died .";
      }
      if(this.jxb == 1)
      {
         zczk2.text = zczk.text;
         zczk.text = "- The colossus of Eastan fell .";
      }
      else if(this.knight == 1)
      {
         zczk2.text = zczk.text;
         zczk.text = "- A knight of Eastan died .";
      }
      else
      {
         zczk2.text = zczk.text;
         zczk.text = "- A footman of Eastan died .";
      }
   }
   this.xue.gotoAndStop(int(this.life / this.xxx * 100));
   if(this.attackzl == 1 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goaljl = Math.abs(people[i]._x - this._x);
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.goal = ecastle;
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(this.hitTest(this.goal) && this._currentframe < 28)
      {
         if(Math.abs(this.goal._y - this._y) > 6)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         else
         {
            if(this._x < this.goal._x)
            {
               this.gotoAndPlay(51);
            }
            if(this._x == this.goal._x)
            {
               if(this._y > this.goal._y)
               {
                  this.gotoAndPlay(51);
               }
               if(this._y < this.goal._y)
               {
                  this.gotoAndPlay(88);
               }
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(88);
            }
         }
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.hitTest(this.goal) && Math.abs(this.goal._y - this._y) < 36 && this._currentframe > 28 && this._currentframe < 120)
      {
         k = 1;
         while(k < 15)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 62 or this._currentframe == 91)
         {
            this.goal.life = this.goal.life - (this.attack + Math.random() * this.attack);
         }
         else if(this.goal == ecastle)
         {
            this.goalxs = 0;
            this.goaljl = 650;
            i = 16;
            while(i < 30)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
               {
                  this.gotoAndStop(1);
                  this.goalxs = this.goalxs + 1;
               }
               else if(i == 29 && this.goalxs == 0)
               {
                  this.goal = ecastle;
               }
               i++;
            }
         }
      }
      else
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goaljl = Math.abs(people[i]._x - this._x);
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.goal = ecastle;
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - this._x));
         if(this.goal._x - this._x > 0 && this.hitTest(this.goal) == 0)
         {
            this.dx = Math.cos(this.angle) * this.speed;
            this.dy = Math.sin(this.angle) * this.speed;
         }
         if(this.goal._x - this._x == 0 or this.hitTest(this.goal))
         {
            if(this.goal._y - this._y > 0)
            {
               this.dx = 0;
               this.dy = 1;
            }
            if(this.goal._y - this._y < 0)
            {
               this.dx = 0;
               this.dy = -1;
            }
            if(this.goal._y - this._y == 0)
            {
               this.dx = 0;
               this.dy = 0;
            }
         }
         if(this.goal._x - this._x < 0 && this.hitTest(this.goal) == 0)
         {
            this.dx = (- Math.cos(this.angle)) * this.speed;
            this.dy = (- Math.sin(this.angle)) * this.speed;
         }
         this._x = this._x + this.dx;
         this._y = this._y + this.dy;
      }
   }
   if(this.attackzl == 2 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 80;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goaljl = Math.abs(people[i]._x - this._x);
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.attackzl = 0;
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(this.hitTest(this.goal) && this._currentframe < 28)
      {
         if(Math.abs(this.goal._y - this._y) > 6)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         else
         {
            if(this._x < this.goal._x)
            {
               this.gotoAndPlay(51);
            }
            if(this._x == this.goal._x)
            {
               if(this._y > this.goal._y)
               {
                  this.gotoAndPlay(51);
               }
               if(this._y < this.goal._y)
               {
                  this.gotoAndPlay(88);
               }
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(88);
            }
         }
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.hitTest(this.goal) && Math.abs(this.goal._y - this._y) < 36 && this._currentframe > 28 && this._currentframe < 120)
      {
         k = 1;
         while(k < 15)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 62 or this._currentframe == 91)
         {
            this.goal.life = this.goal.life - (this.attack + Math.random() * this.attack);
         }
         else if(this.goal == ecastle)
         {
            this.goalxs = 0;
            this.goaljl = 80;
            i = 16;
            while(i < 30)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
               {
                  this.gotoAndStop(1);
                  this.goalxs = this.goalxs + 1;
               }
               else if(i == 29 && this.goalxs == 0)
               {
                  this.attackzl = 0;
               }
               i++;
            }
         }
      }
      else
      {
         this.goalxs = 0;
         this.goaljl = 70;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goaljl = Math.abs(people[i]._x - this._x);
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.attackzl = 0;
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - this._x));
         if(this.goal._x - this._x > 0 && this.hitTest(this.goal) == 0)
         {
            this.dx = Math.cos(this.angle) * this.speed;
            this.dy = Math.sin(this.angle) * this.speed;
         }
         if(this.goal._x - this._x == 0 or this.hitTest(this.goal))
         {
            if(this.goal._y - this._y > 0)
            {
               this.dx = 0;
               this.dy = 1;
            }
            if(this.goal._y - this._y < 0)
            {
               this.dx = 0;
               this.dy = -1;
            }
            if(this.goal._y - this._y == 0)
            {
               this.dx = 0;
               this.dy = 0;
            }
         }
         if(this.goal._x - this._x < 0 && this.hitTest(this.goal) == 0)
         {
            this.dx = (- Math.cos(this.angle)) * this.speed;
            this.dy = (- Math.sin(this.angle)) * this.speed;
         }
         this._x = this._x + this.dx;
         this._y = this._y + this.dy;
      }
   }
   if(this.attackzl == 3 && this._currentframe < 120)
   {
      this.goal = estan;
      if(this._currentframe > 28 && this._currentframe < 120)
      {
         this.gotoAndPlay(15);
      }
      if(this._currentframe < 15 && this._currentframe < 120)
      {
         this.gotoAndPlay(15);
      }
      this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - 70 - this._x));
      if(this.goal._x - 70 - this._x > 0)
      {
         this.dx = Math.cos(this.angle) * this.speed;
         this.dy = Math.sin(this.angle) * this.speed;
      }
      if(this.goal._x - 70 - this._x == 0)
      {
         if(Math.abs(this.goal._y - this._y) >= 180)
         {
            if(this.goal._y - this._y > 0)
            {
               this.dx = 0;
               this.dy = 1;
            }
            if(this.goal._y - this._y < 0)
            {
               this.dx = 0;
               this.dy = -1;
            }
         }
      }
      if(this.goal._x - 70 - this._x < 0)
      {
         this.dx = (- Math.cos(this.angle)) * this.speed;
         this.dy = (- Math.sin(this.angle)) * this.speed;
      }
      this._x = this._x + this.dx;
      this._y = this._y + this.dy;
   }
   if(this.attackzl == 0 && this._currentframe < 120)
   {
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k < 16)
         {
            changsx(this,people[k]);
         }
         k++;
      }
      this.goalxs = 0;
      this.goaljl = 80;
      i = 16;
      while(i < 30)
      {
         if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
         {
            this.goaljl = Math.abs(people[i]._x - this._x);
            this.goal = people[i];
            this.attackzl = 2;
            this.goalxs = this.goalxs + 1;
         }
         else if(i == 29 && this.goalxs == 0)
         {
            this.gotoAndStop(1);
         }
         i++;
      }
   }
}
function js2Action()
{
   if(this._currentframe == 146)
   {
      if(this.herojh2 == 1)
      {
         ecastle.h1 = 1;
         hero2xl = 1;
      }
      if(this.herojh22 == 1)
      {
         ecastle.h2 = 1;
         hero22xl = 1;
      }
      if(this.herojh23 == 1)
      {
         ecastle.h3 = 1;
         hero23xl = 1;
      }
      ecastle.unit--;
      ebsxlh.push(this.number);
      this.removeMovieClip();
   }
   if(this.life < this.xxx && this.heroz == 1)
   {
      this.life = this.life + 0.3;
   }
   if(this.life <= 0 && this._currentframe < 120)
   {
      this.gotoAndPlay(120);
      if(this.heroz == 1)
      {
         zczk2.text = zczk.text;
         zczk.text = "- The hero of Beilmen died .";
      }
      else if(this.knight == 1)
      {
         zczk2.text = zczk.text;
         zczk.text = "- A knight of Beilmen died .";
      }
      else
      {
         zczk2.text = zczk.text;
         zczk.text = "- A footman of Beilmen died .";
      }
   }
   this.xue.gotoAndStop(int(this.life / this.xxx * 100));
   if(this.attackzl == 1 && this._currentframe < 120)
   {
      if(this._currentframe !== 1)
      {
         this.gotoAndPlay(this._currentframe);
      }
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         k++;
      }
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 1;
         while(i < 15)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.gotoAndStop(1);
               this.goalxs = this.goalxs + 1;
            }
            else if(i == 14 && this.goalxs == 0)
            {
               this.goal = estan;
            }
            i++;
         }
         if(this._x <= this.goal._x)
         {
            this.gotoAndPlay(2);
         }
         if(this._x > this.goal._x)
         {
            this.gotoAndPlay(15);
         }
      }
      if(this.hitTest(this.goal) && this._currentframe < 28)
      {
         if(Math.abs(this.goal._y - this._y) > 6)
         {
            if(this.goal._y - this._y > 0)
            {
               this._y = this._y + 1;
            }
            else
            {
               this._y = this._y - 1;
            }
         }
         else
         {
            if(this._x < this.goal._x)
            {
               this.gotoAndPlay(51);
            }
            if(this._x == this.goal._x)
            {
               if(this._y > this.goal._y)
               {
                  this.gotoAndPlay(51);
               }
               if(this._y < this.goal._y)
               {
                  this.gotoAndPlay(88);
               }
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(88);
            }
         }
      }
      if(this.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndPlay(120);
      }
      if(this.goal.life <= 0 && this._currentframe < 120)
      {
         this.gotoAndStop(1);
      }
      if(this.hitTest(this.goal) && Math.abs(this.goal._y - this._y) < 36 && this._currentframe > 28 && this._currentframe < 120)
      {
         k = 16;
         while(k < 30)
         {
            changsx(this,people[k]);
            k++;
         }
         if(this._currentframe == 62 or this._currentframe == 91)
         {
            this.goal.life = this.goal.life - (this.attack + Math.random() * this.attack);
         }
         else if(this.goal == estan)
         {
            this.goalxs = 0;
            this.goaljl = 650;
            i = 1;
            while(i < 15)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
               {
                  this.goaljl = Math.abs(people[i]._x - this._x);
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
               }
               else if(i == 14 && this.goalxs == 0)
               {
                  this.goal = estan;
               }
               i++;
            }
         }
      }
      else
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 1;
         while(i < 15)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goaljl = Math.abs(people[i]._x - this._x);
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
            }
            else if(i == 14 && this.goalxs == 0)
            {
               this.goal = estan;
            }
            i++;
         }
         if(this._currentframe > 28 && this._currentframe < 120)
         {
            if(this._x <= this.goal._x)
            {
               this.gotoAndPlay(2);
            }
            if(this._x > this.goal._x)
            {
               this.gotoAndPlay(15);
            }
         }
         this.angle = Math.atan((this.goal._y - this._y) / (this.goal._x - this._x));
         if(this.goal._x - this._x > 0 && this.hitTest(this.goal) == 0)
         {
            this.dx = Math.cos(this.angle) * this.speed;
            this.dy = Math.sin(this.angle) * this.speed;
         }
         if(this.goal._x - this._x == 0 or this.hitTest(this.goal))
         {
            if(this.goal._y - this._y > 0)
            {
               this.dx = 0;
               this.dy = 1;
            }
            if(this.goal._y - this._y < 0)
            {
               this.dx = 0;
               this.dy = -1;
            }
            if(this.goal._y - this._y == 0)
            {
               this.dx = 0;
               this.dy = 0;
            }
         }
         if(this.goal._x - this._x < 0 && this.hitTest(this.goal) == 0)
         {
            this.dx = (- Math.cos(this.angle)) * this.speed;
            this.dy = (- Math.sin(this.angle)) * this.speed;
         }
         this._x = this._x + this.dx;
         this._y = this._y + this.dy;
      }
   }
   if(this.attackzl == 0)
   {
      k = 0;
      while(k < 30)
      {
         changlayer(this,people[k]);
         if(k >= 16)
         {
            changsx(this,people[k]);
         }
         k++;
      }
   }
}
dwsxjg.gotoAndStop(61);
hzbzjg.gotoAndStop(61);
lordan._visible = heroan._visible = jsan._visible = hqsan._visible = jydan._visible = fsan._visible = xdsan._visible = hpan._visible = qsan._visible = jxban._visible = estanxlan._visible = estanxlan2._visible = false;
ecb._visible = ebl._visible = eby._visible = ept._visible = est._visible = ejt._visible = efc._visible = egc._visible = etjp._visible = false;
cb2._visible = bl2._visible = by2._visible = ecpt._visible = st2._visible = jt2._visible = fc2._visible = gc2._visible = tjp2._visible = ecastlesl._visible = false;
cban._visible = ptan._visible = stan._visible = jtan._visible = gcan._visible = tman._visible = ysgjan._visible = ljsan._visible = jssjan._visible = hqssjan._visible = xdssjan._visible = qssjan._visible = wyjan._visible = gjsan._visible = false;
bsxlh = new Array(2,3,4,5,6,7,8,9,10,11,12,13);
ebsxlh = new Array(17,18,19,20,21,22,23,24,25,26,27,28,29);
jsdefense = 0;
hqsattack = 0;
xdscure = 0;
qsdefense = 0;
qsattack = 0;
jsdefense2 = 0;
hqsattack2 = 0;
xdscure2 = 0;
qsdefense2 = 0;
qsattack2 = 0;
peoplenm = 1;
ecastle.life = ecastle.xxx = 1600;
estan.gold = 150;
estan.life = estan.xxx = 1600;
estan.unit = 0;
estan.uxxx = 4;
estan.flame = 0;
ecastle.unit = 0;
ecastle.uxxx = 4;
wyj = 0;
wyj2 = 0;
sl = 0.4;
hpxs = 2;
xdsxs = 2;
people.attachMovie("flags",14,peoplenm);
people[14]._y = 330;
people[14]._x = 265;
people[14].life = -1;
people[14].number = 14;
peoplenm++;
createjs = function()
{
   var _loc1_ = bsxlh.pop();
   people.attachMovie("js",_loc1_,peoplenm);
   people[_loc1_]._y = 250;
   people[_loc1_]._x = 200;
   people[_loc1_].onEnterFrame = jsAction;
   people[_loc1_].life = people[_loc1_].xxx = 280 + jsdefense;
   people[_loc1_].speed = 1.6 + Math.random() / 5;
   people[_loc1_].attack = 5;
   people[_loc1_].jlife = 1;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   peoplenm++;
};
createjxb = function()
{
   var _loc1_ = bsxlh.pop();
   people.attachMovie("jxb",_loc1_,peoplenm);
   people[_loc1_]._y = 252;
   people[_loc1_]._x = 200;
   people[_loc1_].onEnterFrame = jsAction;
   people[_loc1_].life = people[_loc1_].xxx = 1200;
   people[_loc1_].speed = 2;
   people[_loc1_].attack = 16;
   people[_loc1_].jlife = 2;
   people[_loc1_].attackzl = 0;
   people[_loc1_].jxb = 1;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 1;
   peoplenm++;
};
createhqs = function()
{
   var _loc1_ = bsxlh.pop();
   people.attachMovie("hqs",_loc1_,peoplenm);
   people[_loc1_]._y = 257;
   people[_loc1_]._x = 156;
   people[_loc1_].onEnterFrame = hqsAction;
   people[_loc1_].life = people[_loc1_].xxx = 180;
   people[_loc1_].speed = 1.3 + Math.random() / 5;
   people[_loc1_].attack = 9 + hqsattack;
   people[_loc1_].jlife = 1;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   peoplenm++;
};
createfs = function()
{
   var _loc1_ = bsxlh.pop();
   people.attachMovie("fs",_loc1_,peoplenm);
   people[_loc1_]._y = 220;
   people[_loc1_]._x = 170;
   people[_loc1_].onEnterFrame = fsAction;
   people[_loc1_].life = people[_loc1_].xxx = 210;
   people[_loc1_].speed = 1.3 + Math.random() / 5;
   people[_loc1_].attack = 19;
   people[_loc1_].jlife = 1;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   peoplenm++;
};
createjyd = function()
{
   var _loc1_ = bsxlh.pop();
   people.attachMovie("jyd",_loc1_,peoplenm);
   people[_loc1_]._y = 260;
   people[_loc1_]._x = 160;
   people[_loc1_].onEnterFrame = jydAction;
   people[_loc1_].life = people[_loc1_].xxx = 150;
   people[_loc1_].speed = 1.3 + Math.random() / 5;
   people[_loc1_].jlife = 1;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   peoplenm++;
};
createhero = function()
{
   var _loc1_ = bsxlh.pop();
   people.attachMovie("hero",_loc1_,peoplenm);
   people[_loc1_]._y = 242;
   people[_loc1_]._x = 190;
   people[_loc1_].onEnterFrame = jsAction;
   people[_loc1_].life = people[_loc1_].xxx = 650;
   people[_loc1_].speed = 1.5 + Math.random() / 5;
   people[_loc1_].attack = 13;
   people[_loc1_].jlife = 1;
   people[_loc1_].heroz = 1;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   peoplenm++;
};
createhero2 = function()
{
   var _loc1_ = ebsxlh.pop();
   people.attachMovie("hero2",_loc1_,peoplenm);
   people[_loc1_]._y = 354;
   people[_loc1_]._x = 440;
   people[_loc1_].onEnterFrame = js2Action;
   people[_loc1_].life = people[_loc1_].xxx = 780;
   people[_loc1_].speed = 1.5 + Math.random() / 5;
   people[_loc1_].attack = 12;
   people[_loc1_].jlife = 0;
   people[_loc1_].heroz = 1;
   people[_loc1_].herojh2 = 1;
   ecastle.h1 = 0;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   ecastle.unit = ecastle.unit + 1;
   peoplenm++;
};
createhero22 = function()
{
   var _loc1_ = ebsxlh.pop();
   people.attachMovie("hero22",_loc1_,peoplenm);
   people[_loc1_]._y = 344;
   people[_loc1_]._x = 435;
   people[_loc1_].onEnterFrame = js2Action;
   people[_loc1_].life = people[_loc1_].xxx = 620;
   people[_loc1_].speed = 1.8;
   people[_loc1_].attack = 13;
   people[_loc1_].jlife = 0;
   people[_loc1_].heroz = 1;
   people[_loc1_].herojh22 = 1;
   ecastle.h2 = 0;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   ecastle.unit = ecastle.unit + 1;
   peoplenm++;
};
createxds = function()
{
   var _loc1_ = bsxlh.pop();
   people.attachMovie("xds",_loc1_,peoplenm);
   people[_loc1_]._y = 194;
   people[_loc1_]._x = 154;
   people[_loc1_].onEnterFrame = xdsAction;
   people[_loc1_].life = people[_loc1_].xxx = 120;
   people[_loc1_].speed = 1.3 + Math.random() / 5;
   people[_loc1_].cure = 15 + xdscure;
   people[_loc1_].jlife = 2;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   peoplenm++;
};
createhqs2 = function()
{
   var _loc1_ = ebsxlh.pop();
   people.attachMovie("hqs2",_loc1_,peoplenm);
   people[_loc1_]._y = 330;
   people[_loc1_]._x = 480;
   people[_loc1_].onEnterFrame = hqs2Action;
   people[_loc1_].life = people[_loc1_].xxx = 180;
   people[_loc1_].speed = 1.3 + Math.random() / 5;
   people[_loc1_].attack = 9 + hqsattack2;
   people[_loc1_].jlife = 0;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   ecastle.unit = ecastle.unit + 1;
   peoplenm++;
};
createfs2 = function()
{
   var _loc1_ = ebsxlh.pop();
   people.attachMovie("fs2",_loc1_,peoplenm);
   people[_loc1_]._y = 365;
   people[_loc1_]._x = 470;
   people[_loc1_].onEnterFrame = fs2Action;
   people[_loc1_].life = people[_loc1_].xxx = 180;
   people[_loc1_].speed = 1.3 + Math.random() / 5;
   people[_loc1_].attack = 19;
   people[_loc1_].jlife = 0;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   ecastle.unit = ecastle.unit + 1;
   peoplenm++;
};
createjyd2 = function()
{
   var _loc1_ = ebsxlh.pop();
   people.attachMovie("jyd2",_loc1_,peoplenm);
   people[_loc1_]._y = 350;
   people[_loc1_]._x = 470;
   people[_loc1_].onEnterFrame = jyd2Action;
   people[_loc1_].life = people[_loc1_].xxx = 150;
   people[_loc1_].speed = 1.3 + Math.random() / 5;
   people[_loc1_].jlife = 0;
   ecastle.h4 = 0;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   ecastle.unit = ecastle.unit + 1;
   peoplenm++;
};
createjs2 = function()
{
   var _loc1_ = ebsxlh.pop();
   people.attachMovie("js2",_loc1_,peoplenm);
   people[_loc1_]._y = 332;
   people[_loc1_]._x = 450;
   people[_loc1_].onEnterFrame = js2Action;
   people[_loc1_].life = people[_loc1_].xxx = 280 + jsdefense2;
   people[_loc1_].speed = 1.6 + Math.random() / 5;
   people[_loc1_].attack = 5;
   people[_loc1_].jlife = 0;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   ecastle.unit = ecastle.unit + 1;
   peoplenm++;
};
createhero23 = function()
{
   var _loc1_ = ebsxlh.pop();
   people.attachMovie("hero23",_loc1_,peoplenm);
   people[_loc1_]._y = 354;
   people[_loc1_]._x = 435;
   people[_loc1_].onEnterFrame = js2Action;
   people[_loc1_].life = people[_loc1_].xxx = 800;
   people[_loc1_].speed = 2.4;
   people[_loc1_].attack = 14;
   people[_loc1_].jlife = 0;
   people[_loc1_].heroz = 1;
   people[_loc1_].herojh23 = 1;
   ecastle.h3 = 0;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   ecastle.unit = ecastle.unit + 1;
   peoplenm++;
};
createxds2 = function()
{
   var _loc1_ = ebsxlh.pop();
   people.attachMovie("xds2",_loc1_,peoplenm);
   people[_loc1_]._y = 379;
   people[_loc1_]._x = 480;
   people[_loc1_].onEnterFrame = xds2Action;
   people[_loc1_].life = people[_loc1_].xxx = 120;
   people[_loc1_].speed = 1.3 + Math.random() / 5;
   people[_loc1_].cure = 15 + xdscure2;
   people[_loc1_].jlife = 2;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   ecastle.unit = ecastle.unit + 1;
   peoplenm++;
};
createhp2 = function()
{
   var _loc1_ = ebsxlh.pop();
   people.attachMovie("hp2",_loc1_,peoplenm);
   people[_loc1_]._y = 354;
   people[_loc1_]._x = 470;
   people[_loc1_].onEnterFrame = hp2Action;
   people[_loc1_].life = people[_loc1_].xxx = 80;
   people[_loc1_].speed = 0.6 + Math.random() / 5;
   people[_loc1_].attack = 24;
   people[_loc1_].jlife = 2;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 1;
   ecastle.unit = ecastle.unit + 1;
   peoplenm++;
};
createhp = function()
{
   var _loc1_ = bsxlh.pop();
   people.attachMovie("hp",_loc1_,peoplenm);
   people[_loc1_]._y = 224;
   people[_loc1_]._x = 156;
   people[_loc1_].onEnterFrame = hpAction;
   people[_loc1_].life = people[_loc1_].xxx = 80;
   people[_loc1_].speed = 0.6 + Math.random() / 5;
   people[_loc1_].attack = 24;
   people[_loc1_].jlife = 2;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 1;
   peoplenm++;
};
createqs = function()
{
   var _loc1_ = bsxlh.pop();
   people.attachMovie("qs",_loc1_,peoplenm);
   people[_loc1_]._y = 217;
   people[_loc1_]._x = 188;
   people[_loc1_].onEnterFrame = jsAction;
   people[_loc1_].life = people[_loc1_].xxx = 480 + qsdefense;
   people[_loc1_].speed = 2.5 - Math.random() / 5;
   people[_loc1_].attack = 9 + qsattack;
   people[_loc1_].jlife = 1;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].knight = 1;
   people[_loc1_].jx = 0;
   peoplenm++;
};
createhero12 = function()
{
   var _loc1_ = bsxlh.pop();
   people.attachMovie("hero12",_loc1_,peoplenm);
   people[_loc1_]._y = 267;
   people[_loc1_]._x = 156;
   people[_loc1_].onEnterFrame = hqsAction;
   people[_loc1_].life = people[_loc1_].xxx = 480;
   people[_loc1_].speed = 1.5;
   people[_loc1_].attack = 17;
   people[_loc1_].jlife = 1;
   people[_loc1_].heroz = 1;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].jx = 0;
   peoplenm++;
};
createqs2 = function()
{
   var _loc1_ = ebsxlh.pop();
   people.attachMovie("qs2",_loc1_,peoplenm);
   people[_loc1_]._y = 365;
   people[_loc1_]._x = 445;
   people[_loc1_].onEnterFrame = js2Action;
   people[_loc1_].life = people[_loc1_].xxx = 480 + qsdefense2;
   people[_loc1_].speed = 2.5 - Math.random() / 5;
   people[_loc1_].attack = 9 + qsattack2;
   people[_loc1_].jlife = 0;
   people[_loc1_].attackzl = 0;
   people[_loc1_].number = _loc1_;
   people[_loc1_].knight = 1;
   people[_loc1_].jx = 0;
   ecastle.unit = ecastle.unit + 1;
   peoplenm++;
};
attackan.onRelease = function()
{
   k = 0;
   while(k < 15)
   {
      people[k].attackzl = 1;
      people[k].goal = ecastle;
      if(people[k]._currentframe < 51)
      {
         people[k].gotoAndPlay(2);
      }
      k++;
   }
};
backan.onRelease = function()
{
   k = 0;
   while(k < 15)
   {
      people[k].attackzl = 3;
      people[k].goal = estan;
      k++;
   }
};
estanxlan.onRelease = function()
{
   estansl._visible = true;
};
estanxlan2.onRelease = function()
{
   estansl._visible = false;
};
lordan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
lordan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
lordan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 225)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.unit == estan.uxxx)
      {
         dwsxjg.gotoAndPlay(1);
      }
      if(estan.gold >= 225 && estan.unit < estan.uxxx)
      {
         estan.gold = estan.gold - 225;
         this.gotoAndPlay(3);
         estan.unit = estan.unit + 1;
      }
   }
};
heroan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
heroan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
heroan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 210)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.unit == estan.uxxx)
      {
         dwsxjg.gotoAndPlay(1);
      }
      if(estan.gold >= 210 && estan.unit < estan.uxxx)
      {
         estan.gold = estan.gold - 210;
         this.gotoAndPlay(3);
         estan.unit = estan.unit + 1;
      }
   }
};
jxban.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
jxban.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
jxban.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 320)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.unit == estan.uxxx)
      {
         dwsxjg.gotoAndPlay(1);
      }
      if(estan.gold >= 320 && estan.unit < estan.uxxx)
      {
         estan.gold = estan.gold - 320;
         this.gotoAndPlay(3);
         estan.unit = estan.unit + 1;
      }
   }
};
jsan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
jsan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
jsan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 70)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.unit == estan.uxxx)
      {
         dwsxjg.gotoAndPlay(1);
      }
      if(estan.gold >= 70 && estan.unit < estan.uxxx)
      {
         estan.gold = estan.gold - 70;
         jsan.gotoAndPlay(3);
         estan.unit = estan.unit + 1;
      }
   }
};
hqsan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
hqsan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
hqsan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 90)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.unit == estan.uxxx)
      {
         dwsxjg.gotoAndPlay(1);
      }
      if(estan.gold >= 90 && estan.unit < estan.uxxx)
      {
         estan.gold = estan.gold - 90;
         this.gotoAndPlay(3);
         estan.unit = estan.unit + 1;
      }
   }
};
fsan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
fsan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
fsan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 125)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.unit == estan.uxxx)
      {
         dwsxjg.gotoAndPlay(1);
      }
      if(estan.gold >= 125 && estan.unit < estan.uxxx)
      {
         estan.gold = estan.gold - 125;
         this.gotoAndPlay(3);
         estan.unit = estan.unit + 1;
      }
   }
};
xdsan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
xdsan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
xdsan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 80)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.unit == estan.uxxx)
      {
         dwsxjg.gotoAndPlay(1);
      }
      if(estan.gold >= 80 && estan.unit < estan.uxxx)
      {
         estan.gold = estan.gold - 80;
         this.gotoAndPlay(3);
         estan.unit = estan.unit + 1;
      }
   }
};
jydan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
jydan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
jydan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 95)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.unit == estan.uxxx)
      {
         dwsxjg.gotoAndPlay(1);
      }
      if(estan.gold >= 95 && estan.unit < estan.uxxx)
      {
         estan.gold = estan.gold - 95;
         jydan.gotoAndPlay(3);
         estan.unit = estan.unit + 1;
      }
   }
};
hpan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
hpan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
hpan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 120)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.unit == estan.uxxx)
      {
         dwsxjg.gotoAndPlay(1);
      }
      if(estan.gold >= 120 && estan.unit < estan.uxxx)
      {
         this.gotoAndPlay(3);
         estan.gold = estan.gold - 120;
         estan.unit = estan.unit + 1;
      }
   }
};
qsan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
qsan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
qsan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 140)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.unit == estan.uxxx)
      {
         dwsxjg.gotoAndPlay(1);
      }
      if(estan.gold >= 140 && estan.unit < estan.uxxx)
      {
         estan.gold = estan.gold - 140;
         this.gotoAndPlay(3);
         estan.unit = estan.unit + 1;
      }
   }
};
byan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
byan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
byan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 120)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 120)
      {
         estan.gold = estan.gold - 120;
         this.gotoAndPlay(3);
      }
   }
};
jtan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
jtan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
jtan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 210)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 210)
      {
         estan.gold = estan.gold - 210;
         this.gotoAndPlay(3);
      }
   }
};
gcan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
gcan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
gcan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 230)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 230)
      {
         estan.gold = estan.gold - 230;
         this.gotoAndPlay(3);
      }
   }
};
ptan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
ptan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
ptan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 160)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 160)
      {
         estan.gold = estan.gold - 160;
         this.gotoAndPlay(3);
      }
   }
};
blan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
blan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
blan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 500)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 500)
      {
         estan.gold = estan.gold - 500;
         this.gotoAndPlay(3);
      }
   }
};
stan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
stan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
stan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 300)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 300)
      {
         estan.gold = estan.gold - 300;
         this.gotoAndPlay(3);
      }
   }
};
cban.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
cban.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
cban.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 900)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 900)
      {
         estan.gold = estan.gold - 900;
         this.gotoAndPlay(3);
      }
   }
};
jssjan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
jssjan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
jssjan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 100)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 100)
      {
         estan.gold = estan.gold - 100;
         this.gotoAndPlay(3);
      }
   }
};
xdssjan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
xdssjan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
xdssjan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 120)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 120)
      {
         estan.gold = estan.gold - 120;
         this.gotoAndPlay(3);
      }
   }
};
hqssjan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
hqssjan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
hqssjan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 180)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 180)
      {
         estan.gold = estan.gold - 180;
         this.gotoAndPlay(3);
      }
   }
};
qssjan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
qssjan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
qssjan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 180)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 180)
      {
         estan.gold = estan.gold - 180;
         this.gotoAndPlay(3);
      }
   }
};
tjpan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
tjpan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
tjpan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 220)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 220)
      {
         estan.gold = estan.gold - 220;
         this.gotoAndPlay(3);
      }
   }
};
tman.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
tman.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
tman.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 130)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 130)
      {
         estan.gold = estan.gold - 130;
         this.gotoAndPlay(3);
      }
   }
};
wyjan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
wyjan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
wyjan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 150)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 150)
      {
         estan.gold = estan.gold - 150;
         this.gotoAndPlay(3);
      }
   }
};
gjsan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
gjsan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
gjsan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 785)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 785)
      {
         estan.gold = estan.gold - 785;
         this.gotoAndPlay(3);
      }
   }
};
fcan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
fcan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
fcan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 150)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 150)
      {
         estan.gold = estan.gold - 150;
         this.gotoAndPlay(3);
      }
   }
};
ysgjan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
ysgjan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
ysgjan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 260)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 260)
      {
         estan.gold = estan.gold - 260;
         this.gotoAndPlay(3);
      }
   }
};
ljsan.onRollOver = function()
{
   if(this._visible == true && this._currentframe == 1)
   {
      this.gotoAndStop(2);
      dwsxjg.gotoAndStop(61);
      hzbzjg.gotoAndStop(61);
   }
};
ljsan.onRollOut = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      this.gotoAndStop(1);
   }
};
ljsan.onRelease = function()
{
   if(this._visible == true && this._currentframe == 2)
   {
      if(estan.gold < 500)
      {
         hzbzjg.gotoAndPlay(1);
      }
      if(estan.gold >= 500)
      {
         estan.gold = estan.gold - 500;
         this.gotoAndPlay(3);
      }
   }
};
xxxxx = 0;
js2xl = 0;
jsxst = 0;
hqs2xl = 0;
hqsxst = 0;
xds2xl = 0;
hp2xl = 0;
fs2xl = 0;
qs2xl = 0;
attack2xl = 0;
hero2xl = 0;
hero22xl = 0;
hero3xl = 0;
hero23xl = 0;
ecastle.flame = 0;
ecastle.h1 = 1;
ecastle.h2 = 1;
ecastle.h3 = 1;
ecastle.h4 = 1;
ecastle.gold = 250;
ecastle.onEnterFrame = function()
{
   this.gold = this.gold + ecsl;
   egolds.text = int(this.gold);
   ecastlelife.text = int(this.life);
   ecastlexxx.text = int(this.xxx);
   if(this.life <= 500 && this.flame == 0)
   {
      this.flame = 1;
      flamelayer.attachMovie("flame",5,5);
      flamelayer[5]._x = 559;
      flamelayer[5]._y = 320;
      flamelayer.attachMovie("flame",6,6);
      flamelayer[6]._x = 509;
      flamelayer[6]._y = 383;
      flamelayer.attachMovie("flame",7,7);
      flamelayer[7]._x = 627;
      flamelayer[7]._y = 373;
      flamelayer.attachMovie("flame",8,8);
      flamelayer[8]._x = 535;
      flamelayer[8]._y = 418;
   }
   if(this.life > 500 && this.flame == 1)
   {
      this.flame = 0;
      i = 4;
      while(i < 9)
      {
         flamelayer[i].removeMovieClip();
         i++;
      }
   }
   if(this.life <= 0)
   {
      hpxg.removeMovieClip();
      xdsxg.removeMovieClip();
      people.removeMovieClip();
      _root.gotoAndStop(13);
   }
   if(this.goalxs == 1)
   {
      k = 16;
      while(k < 30)
      {
         people[k].attackzl = 1;
         k++;
      }
   }
   this.goalxs = 0;
   i = 1;
   while(i < 15)
   {
      if(people[i]._x > 320 && people[i]._currentframe < 120)
      {
         this.goalxs = 1;
      }
      i++;
   }
   i = 16;
   while(i < 30)
   {
      if(people[i].attackzl == 0 && people[i].life < people[i].xxx)
      {
         this.goalxs = 1;
      }
      i++;
   }
   if(xxxxx < 420 * r)
   {
      if(xxxxx == 250)
      {
         by2._visible = true;
      }
      if(xxxxx == 422)
      {
         createjs2();
         this.gold = this.gold - 60;
      }
   }
   if(xxxxx > 420 * r && xxxxx < 740 * r)
   {
      if(js2xl == 0 && ecastle.unit < ecastle.uxxx)
      {
         createjs2();
         this.gold = this.gold - 60;
      }
      if(attack2xl == 0)
      {
         k = 16;
         while(k < 30)
         {
            people[k].attackzl = 1;
            k++;
         }
      }
      if(xxxxx == int(739 * r))
      {
         tjp2._visible = true;
         ecastle.uxxx = ecastle.uxxx + 1;
      }
   }
   if(xxxxx > 740 * r && xxxxx < 2280 * r)
   {
      if(js2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 60)
      {
         createjs2();
         this.gold = this.gold - 60;
      }
      if(hqs2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 90)
      {
         createhqs2();
         this.gold = this.gold - 90;
      }
      if(attack2xl == 0)
      {
         k = 16;
         while(k < 30)
         {
            people[k].attackzl = 1;
            k++;
         }
      }
      if(xxxxx == int(940 * r))
      {
         jsxst = 60;
      }
      if(xxxxx == int(1000 * r))
      {
         fc2._visible = true;
      }
      if(xxxxx == int(1800 * r))
      {
         bl2._visible = true;
         ecastle.life = ecastle.life + 600;
         ecastle.xxx = ecastle.xxx + 600;
         ecastle.uxxx = ecastle.uxxx + 2;
      }
      if(xxxxx == int(1200 * r))
      {
         ecpt._visible = true;
         wyj2 = 70;
      }
      if(xxxxx == int(1980 * r))
      {
         wyj2 = 70;
      }
      if(xxxxx == int(2199 * r))
      {
         ecsl = ecsl + 0.2;
         st2._visible = true;
         hero2xl = 1;
         hqsxst = 90;
      }
   }
   if(xxxxx > 2280 * r && xxxxx < 2980 * r)
   {
      if(hero2xl == 0 && ecastle.h1 == 1 && ecastle.unit < ecastle.uxxx && this.gold > 210)
      {
         createhero2();
         this.gold = this.gold - 210;
      }
      if(js2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 60)
      {
         createjs2();
         this.gold = this.gold - 60;
      }
      if(hqs2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 90)
      {
         createhqs2();
         this.gold = this.gold - 90;
      }
      if(attack2xl == 0)
      {
         k = 16;
         while(k < 30)
         {
            people[k].attackzl = 1;
            k++;
         }
      }
      if(xxxxx == int(2500 * r))
      {
         jsdefense2 = 40;
      }
      if(xxxxx == int(2899 * r))
      {
         ecsl = ecsl + 0.2;
         xds2xl = 1;
         jt2._visible = true;
      }
   }
   if(xxxxx > 2980 * r && xxxxx < 3500 * r)
   {
      if(hero2xl == 0 && ecastle.h1 == 1 && ecastle.unit < ecastle.uxxx && this.gold > 210)
      {
         createhero2();
         this.gold = this.gold - 210;
      }
      if(js2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 60)
      {
         createjs2();
         this.gold = this.gold - 60;
      }
      if(hqs2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 90)
      {
         createhqs2();
         this.gold = this.gold - 90;
      }
      if(xds2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 100)
      {
         createxds2();
         this.gold = this.gold - 100;
      }
      if(attack2xl == 0)
      {
         k = 16;
         while(k < 30)
         {
            people[k].attackzl = 1;
            k++;
         }
      }
      if(xxxxx == int(3100 * r))
      {
         hqsattack2 = 3;
      }
      if(xxxxx == int(3400 * r))
      {
         gc2._visible = true;
         ecastle.uxxx = ecastle.uxxx + 1;
      }
   }
   if(xxxxx > 3500 * r && xxxxx < 4700 * r)
   {
      k = 16;
      while(k < 30)
      {
         if(people[k].heroz == 1)
         {
            k = 16;
            while(k < 30)
            {
               people[k].attackzl = 1;
               k++;
            }
         }
         k++;
      }
      if(hero3xl == 0 && ecastle.h4 == 1 && ecastle.unit < ecastle.uxxx && this.gold > 95)
      {
         createjyd2();
         this.gold = this.gold - 95;
      }
      if(hero2xl == 0 && ecastle.h1 == 1 && ecastle.unit < ecastle.uxxx && this.gold > 210)
      {
         createhero2();
         this.gold = this.gold - 210;
      }
      if(xds2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 100)
      {
         createxds2();
         this.gold = this.gold - 100;
      }
      if(hp2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 120)
      {
         createhp2();
         this.gold = this.gold - 120;
      }
      if(js2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 60)
      {
         createjs2();
         this.gold = this.gold - 60;
      }
      if(hqs2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 90)
      {
         createhqs2();
         this.gold = this.gold - 90;
      }
      if(attack2xl == 0)
      {
         k = 16;
         while(k < 30)
         {
            people[k].attackzl = 1;
            k++;
         }
      }
      if(xxxxx == int(4050 * r))
      {
         ecsl = ecsl + 0.2;
         ecastle.life = ecastle.life + 1200;
         ecastle.xxx = ecastle.xxx + 1200;
      }
      if(xxxxx == int(4200 * r))
      {
         xdscure2 = 4;
      }
      if(xxxxx == int(4500 * r))
      {
         cb2._visible = true;
         hero22xl = 1;
         qs2xl = 1;
         fs2xl = 1;
         ecastle.life = ecastle.life + 600;
         ecastle.xxx = ecastle.xxx + 600;
         ecastle.uxxx = ecastle.uxxx + 3;
      }
   }
   if(xxxxx > 4700 * r && xxxxx < 5500 * r)
   {
      if(hero22xl == 0 && ecastle.h2 == 1 && ecastle.unit < ecastle.uxxx && this.gold > 200)
      {
         createhero22();
         this.gold = this.gold - 200;
      }
      if(qs2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 140)
      {
         createqs2();
         this.gold = this.gold - 140;
      }
      if(hero2xl == 0 && ecastle.h1 == 1 && ecastle.unit < ecastle.uxxx && this.gold > 210)
      {
         createhero2();
         this.gold = this.gold - 210;
      }
      if(fs2xl == 0 && ecastle.unit < ecastle.uxxx)
      {
         createfs2();
      }
      if(hero3xl == 0 && ecastle.h4 == 1 && ecastle.unit < ecastle.uxxx && this.gold > 95)
      {
         createjyd2();
         this.gold = this.gold - 95;
      }
      if(xds2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 100)
      {
         createxds2();
         this.gold = this.gold - 100;
      }
      if(hp2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 120)
      {
         createhp2();
         this.gold = this.gold - 120;
      }
      if(hqs2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 90)
      {
         createhqs2();
         this.gold = this.gold - 90;
      }
      if(js2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 60)
      {
         createjs2();
         this.gold = this.gold - 60;
      }
      if(attack2xl == 0)
      {
         k = 16;
         while(k < 30)
         {
            people[k].attackzl = 1;
            k++;
         }
      }
      if(xxxxx == int(5100 * r))
      {
         qsdefense2 = 50;
         qsattack2 = 1;
         hero23xl = 1;
      }
   }
   if(xxxxx > 5500 * r)
   {
      k = 16;
      while(k < 30)
      {
         if(people[k].heroz == 1)
         {
            k = 16;
            while(k < 30)
            {
               people[k].attackzl = 1;
               k++;
            }
         }
         k++;
      }
      if(hero23xl == 0 && ecastle.h3 == 1 && ecastle.unit < ecastle.uxxx && this.gold > 200)
      {
         createhero23();
         this.gold = this.gold - 200;
      }
      if(hero22xl == 0 && ecastle.h2 == 1 && this.gold > 200)
      {
         createhero22();
         this.gold = this.gold - 200;
      }
      if(qs2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 140)
      {
         createqs2();
         this.gold = this.gold - 140;
      }
      if(hero2xl == 0 && ecastle.h1 == 1 && ecastle.unit < ecastle.uxxx && this.gold > 210)
      {
         createhero2();
         this.gold = this.gold - 210;
      }
      if(hero3xl == 0 && ecastle.h4 == 1 && ecastle.unit < ecastle.uxxx && this.gold > 95)
      {
         createjyd2();
         this.gold = this.gold - 95;
      }
      if(xds2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 100)
      {
         createxds2();
         this.gold = this.gold - 100;
      }
      if(hp2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 120)
      {
         createhp2();
         this.gold = this.gold - 120;
      }
      if(fs2xl == 0 && ecastle.unit < ecastle.uxxx)
      {
         createfs2();
      }
      if(hqs2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 90)
      {
         createhqs2();
         this.gold = this.gold - 90;
      }
      if(js2xl == 0 && ecastle.unit < ecastle.uxxx && this.gold > 60)
      {
         createjs2();
         this.gold = this.gold - 60;
      }
      if(attack2xl == 0)
      {
         k = 16;
         while(k < 30)
         {
            people[k].attackzl = 1;
            k++;
         }
      }
      if(xxxxx == int(4900 * r))
      {
         qsdefense2 = 50;
         qsattack2 = 1;
      }
   }
   if(this.life < this.xxx && tjp2._visible == true)
   {
      this.life = this.life + 0.4;
      ecastlesl._visible = true;
   }
   if(this.life >= this.xxx)
   {
      ecastlesl._visible = false;
   }
   xxxxx++;
   js2xl++;
   hqs2xl++;
   xds2xl++;
   hp2xl++;
   fs2xl++;
   qs2xl++;
   attack2xl++;
   hero2xl++;
   hero22xl++;
   hero23xl++;
   jxb2xl++;
   hero3xl++;
   js2xl = js2xl % int(150 + jsxst + 35 * r);
   hqs2xl = hqs2xl % int(200 + hqsxst + 130 * r);
   xds2xl = xds2xl % int(270 + 100 * r);
   hp2xl = hp2xl % int(350 + 110 * r);
   fs2xl = fs2xl % int(300 + 110 * r);
   qs2xl = qs2xl % int(150 + 130 * r);
   attack2xl = attack2xl % int(900 + 100 * r);
   hero2xl = hero2xl % int(470 * r);
   hero22xl = hero22xl % int(480 * r);
   hero23xl = hero23xl % int(480 * r);
   jxb2xl = jxb2xl % int(620 * r);
   hero3xl = hero3xl % int(120 + 30 * r);
};
estan.onEnterFrame = function()
{
   this.gold = this.gold + sl;
   estangold.text = int(this.gold);
   estanlife.text = int(this.life);
   estanxxx.text = int(this.xxx);
   estanunit.text = this.unit;
   estanuxxx.text = this.uxxx;
   if(heroan._currentframe == 400)
   {
      createhero();
      heroan.gotoAndStop(401);
   }
   if(lordan._currentframe == 465)
   {
      createhero12();
      lordan.gotoAndStop(466);
   }
   if(jxban._currentframe == 460)
   {
      createjxb();
      jxban.gotoAndStop(461);
   }
   if(jsan._currentframe == 180)
   {
      createjs();
      jsan.gotoAndStop(1);
   }
   if(hqsan._currentframe == 250)
   {
      createhqs();
      hqsan.gotoAndStop(1);
   }
   if(fsan._currentframe == 285)
   {
      createfs();
      fsan.gotoAndStop(1);
   }
   if(jydan._currentframe == 250)
   {
      createjyd();
      jydan.gotoAndStop(251);
   }
   if(xdsan._currentframe == 275)
   {
      createxds();
      xdsan.gotoAndStop(1);
   }
   if(hpan._currentframe == 300)
   {
      createhp();
      hpan.gotoAndStop(1);
   }
   if(qsan._currentframe == 220)
   {
      createqs();
      qsan.gotoAndStop(1);
   }
   if(byan._currentframe == 230)
   {
      eby._visible = true;
      jsan._visible = true;
      if(etjp._visible == true)
      {
         jssjan._visible = true;
         hqsan._visible = true;
         hqssjan._visible = true;
      }
      if(ecb._visible == true)
      {
         qsan._visible = true;
      }
   }
   if(jtan._currentframe == 360)
   {
      ejt._visible = true;
      xdsan._visible = true;
      if(ecb._visible == true && etjp._visible == true)
      {
         gjsan._visible = true;
      }
      if(ecb._visible == true)
      {
         fsan._visible = true;
      }
      if(est._visible == true)
      {
         xdssjan._visible = true;
      }
      jtan.gotoAndStop(361);
   }
   if(gcan._currentframe == 420)
   {
      egc._visible = true;
      hpan._visible = true;
      jydan._visible = true;
      estan.uxxx = estan.uxxx + 1;
      if(ebl._visible == true)
      {
         tman._visible = true;
      }
      gcan.gotoAndStop(421);
   }
   if(ptan._currentframe == 490)
   {
      ept._visible = true;
      if(ebl._visible == true)
      {
         wyjan._visible = true;
      }
      ptan.gotoAndStop(491);
   }
   if(blan._currentframe == 600)
   {
      ysgjan._visible = true;
      ebl._visible = true;
      jtan._visible = true;
      gcan._visible = true;
      stan._visible = true;
      cban._visible = true;
      if(ept._visible == true)
      {
         wyjan._visible = true;
      }
      blan.gotoAndStop(601);
      estan.uxxx = estan.uxxx + 2;
      estan.xxx = estan.xxx + 600;
      estan.life = estan.life + 600;
   }
   if(stan._currentframe == 340)
   {
      est._visible = true;
      heroan._visible = true;
      ljsan._visible = true;
      if(ejt._visible == true)
      {
         xdssjan._visible = true;
      }
      stan.gotoAndStop(341);
   }
   if(cban._currentframe == 500)
   {
      ecb._visible = true;
      if(ejt._visible == true && etjp._visible == true)
      {
         gjsan._visible = true;
      }
      if(est._visible == true)
      {
         lordan._visible = true;
      }
      if(eby._visible == true)
      {
         qsan._visible = true;
      }
      if(etjp._visible == true)
      {
         qssjan._visible = true;
      }
      if(ejt._visible == true)
      {
         fsan._visible = true;
      }
      cban.gotoAndStop(501);
      estan.uxxx = estan.uxxx + 2;
      estan.xxx = estan.xxx + 600;
      estan.life = estan.life + 600;
   }
   if(jssjan._currentframe == 450)
   {
      jsdefense = 40;
      jssjan.gotoAndStop(451);
   }
   if(hqssjan._currentframe == 580)
   {
      hqsattack = 3;
      hqssjan.gotoAndStop(581);
   }
   if(xdssjan._currentframe == 550)
   {
      xdscure = 4;
      xdssjan.gotoAndStop(551);
   }
   if(qssjan._currentframe == 510)
   {
      qsdefense = 50;
      qsattack = 1;
      qssjan.gotoAndStop(511);
   }
   if(tjpan._currentframe == 400)
   {
      etjp._visible = true;
      ptan._visible = true;
      estan.uxxx = estan.uxxx + 1;
      estanxlan._visible = true;
      estanxlan2._visible = true;
      if(ejt._visible == true && ecb._visible == true)
      {
         gjsan._visible = true;
      }
      if(eby._visible == true)
      {
         jssjan._visible = true;
         hqsan._visible = true;
         hqssjan._visible = true;
      }
      if(qsan._visible == true)
      {
         qssjan._visible = true;
      }
      tjpan.gotoAndStop(401);
   }
   if(tman._currentframe == 350)
   {
      estan.xxx = estan.xxx + 1200;
      estan.life = estan.life + 1200;
      tman.gotoAndStop(351);
   }
   if(wyjan._currentframe == 400)
   {
      wyj = 70;
      wyjan.gotoAndStop(401);
   }
   if(gjsan._currentframe == 500)
   {
      estan.uxxx = estan.uxxx + 1;
      jxban._visible = true;
      gjsan.gotoAndStop(501);
   }
   if(fcan._currentframe == 560)
   {
      efc._visible = true;
      sl = sl + 0.2;
      fcan.gotoAndStop(561);
   }
   if(ysgjan._currentframe == 580)
   {
      sl = sl + 0.3;
      ysgjan.gotoAndStop(581);
   }
   if(ljsan._currentframe == 600)
   {
      sl = sl + 0.4;
      ljsan.gotoAndStop(601);
   }
   if(this.life <= 500 && this.flame == 0)
   {
      this.flame = 1;
      flamelayer.attachMovie("flame",1,1);
      flamelayer[1]._x = 68;
      flamelayer[1]._y = 158;
      flamelayer.attachMovie("flame",2,2);
      flamelayer[2]._x = 11;
      flamelayer[2]._y = 188;
      flamelayer.attachMovie("flame",3,3);
      flamelayer[3]._x = 122;
      flamelayer[3]._y = 228;
      flamelayer.attachMovie("flame",4,4);
      flamelayer[4]._x = 97;
      flamelayer[4]._y = 240;
   }
   if(this.life > 500 && this.flame == 1)
   {
      this.flame = 0;
      i = 0;
      while(i < 5)
      {
         flamelayer[i].removeMovieClip();
         i++;
      }
   }
   if(this.life <= 0)
   {
      _root.gotoAndStop(10);
   }
   if(this.life < this.xxx && estansl._visible == true)
   {
      this.gold = this.gold - 0.25;
      this.life = this.life + 0.4;
   }
   if(this.life >= this.xxx)
   {
      estansl._visible = false;
   }
};
exue.onEnterFrame = function()
{
   this.gotoAndStop(int(estan.life / estan.xxx * 100));
};
ecxue.onEnterFrame = function()
{
   this.gotoAndStop(int(ecastle.life / ecastle.xxx * 100));
};
ept.onEnterFrame = function()
{
   if(this._visible == true)
   {
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 16;
         while(i < 30)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goaljl = Math.abs(people[i]._x - this._x);
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
            }
            else if(i == 29 && this.goalxs == 0)
            {
               this.goal = ecastle;
            }
            i++;
         }
      }
      if(this.goal._x <= 240 + wyj && this._currentframe == 1)
      {
         this.gotoAndPlay(2);
      }
      if(this.goal._x <= 240 + wyj && this._currentframe > 1)
      {
         if(this._currentframe == 5)
         {
            this.goal.life = this.goal.life - (18 + Math.random() * 12);
         }
         else
         {
            this.goalxs = 0;
            this.goaljl = 650;
            i = 16;
            while(i < 30)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
               {
                  this.goaljl = Math.abs(people[i]._x - this._x);
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
               }
               else if(i == 29 && this.goalxs == 0)
               {
                  this.goal = ecastle;
               }
               i++;
            }
         }
      }
      else
      {
         this.gotoAndStop(1);
      }
   }
};
ecpt.onEnterFrame = function()
{
   if(this._visible == true)
   {
      if(this._currentframe == 1)
      {
         this.goalxs = 0;
         this.goaljl = 650;
         i = 1;
         while(i < 15)
         {
            if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
            {
               this.goaljl = Math.abs(people[i]._x - this._x);
               this.goal = people[i];
               this.goalxs = this.goalxs + 1;
            }
            else if(i == 14 && this.goalxs == 0)
            {
               this.goal = estan;
            }
            i++;
         }
      }
      if(this.goal._x >= 400 - wyj2 && this._currentframe == 1)
      {
         this.gotoAndPlay(2);
      }
      if(this.goal._x >= 400 - wyj2 && this._currentframe > 1)
      {
         if(this._currentframe == 5)
         {
            this.goal.life = this.goal.life - (18 + Math.random() * 12);
         }
         else
         {
            this.goalxs = 0;
            this.goaljl = 650;
            i = 1;
            while(i < 15)
            {
               if(Math.abs(people[i]._x - this._x) < this.goaljl && people[i]._currentframe < 120)
               {
                  this.goaljl = Math.abs(people[i]._x - this._x);
                  this.goal = people[i];
                  this.goalxs = this.goalxs + 1;
               }
               else if(i == 14 && this.goalxs == 0)
               {
                  this.goal = estan;
               }
               i++;
            }
         }
      }
      else
      {
         this.gotoAndStop(1);
      }
   }
};
changlayer = function(x, y)
{
   a = x.getDepth();
   b = y.getDepth();
   if(x._y < y._y && a > b)
   {
      x.swapDepths(y);
   }
   if(x._y > y._y && a < b)
   {
      x.swapDepths(y);
   }
};
changsx = function(x, y)
{
   if(Math.abs(x._y - y._y) < 3 && x.hitTest(y) && y._currentframe > 51)
   {
      if(x.number !== 14 && y.number !== 14)
      {
         if(x._y > y._y)
         {
            x._y = x._y + 3;
            y._y = y._y - 3;
         }
         if(x._y <= y._y)
         {
            x._y = x._y - 3;
            y._y = y._y + 3;
         }
      }
   }
   if(Math.abs(x._y - y._y) < 3 && x.hitTest(y) && y._currentframe == 1)
   {
      if(x.number !== 14 && y.number !== 14)
      {
         if(x._y > y._y)
         {
            x._y = x._y + 3;
            y._y = y._y - 3;
         }
         if(x._y <= y._y)
         {
            x._y = x._y - 3;
            y._y = y._y + 3;
         }
      }
   }
};
