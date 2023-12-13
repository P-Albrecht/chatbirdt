import { Component } from '@angular/core';
import {ActivatedRoute, Params, Router} from "@angular/router";
import { HttpClient } from '@angular/common/http';

import { LlamaService } from './llama.service';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'ChatBirdt';

  constructor(private router: Router, private activatedRoute: ActivatedRoute, private http: HttpClient, private githubService: LlamaService) {
  }

  async ngOnInit() {
    this.router.navigate(['/chat']);

  }
}

